"""Virtual camera framing: compute a smooth crop ROI from ball + players.

Phase 5 — フレーミング統合（放送的＝滑らかさ優先）。

設計の狙い（旧実装の課題と対策）:
  旧 framing は「ボール可視ならボール中心、不可視なら全選手 wide」という
  ハード切替だった。ボールが一瞬でも検出落ちするたびに wide ↔ ball を往復し、
  カメラがガクガク跳ぶ。Phase 5 では以下で放送的な滑らかさを得る。

  1. **加重合成（ハード切替の廃止）**: ターゲット中心は
        center = w_ball · ball + (1 - w_ball) · players_centroid
     とし、ボールの確度で w_ball を連続的に変える。
       - visible（検出）        → w_ball 高（ボール主体）
       - predicted（Kalman 補間）→ w_ball 中（選手集団へ寄せ始める）
       - lost                  → w_ball = 0（選手集団のみ）
     ボールが瞬断しても重みが滑らかに移るだけで、画は跳ばない。

  2. **pan と zoom の分離**: 中心移動（pan）とサイズ変化（zoom）を別 EMA・別
     速度制限で平滑化する。速い pan と緩い zoom を独立に調律でき、ズームの
     ハンチング（伸縮の往復）を抑える。

  3. **デッドゾーン**: ターゲット中心が現フレーム中心の一定範囲内なら pan を
     凍結する。微小なボール揺れにカメラが反応するジッターを断つ。

  4. **先読み（look-ahead）**: ボール速度ベクトルで中心を進行方向へ少し進める。
     高速移動でカメラが後追いになるのを補正する（滑らかさ優先なので控えめ）。

  5. **ゲーム状態ゲーティング**: participation.game_active=False（タイムアウト／
     セット間で court 上の人数が激減）なら、ボールを追わず wide に緩く保持する。
     試合が止まっている間、選手の出入りにカメラが振り回されない。

出力はクロップ座標 ROI のみ（レンダリングはしない）。アスペクト比維持・フレーム
内クランプ・最小ズーム保証は従来どおり。

平滑化は速度制限つき EMA。低 alpha ほど滑らかだが追従が遅い（プリンシパル7:
挙動を観測できるよう、内部状態は debug_framing で可視化できる）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ball_tracker import BallState
from .tracker import Track


@dataclass
class ROI:
    x: int  # top-left column
    y: int  # top-left row
    w: int  # width
    h: int  # height


# ── 平滑化プロファイル（放送的＝滑らかさ優先のデフォルト） ─────────────────────────
# NG 評価 (2026-06) で判明した問題と調律:
#   - ボール左右追従でカメラがスウェーする: _W_BALL_VISIBLE を下げ、
#     選手集団重心をアンカーとして利かせる。
#   - pan 速度が速すぎる（カクつく）: _PAN_ALPHA と _MAX_PAN_FRAC を大幅に下げる。
#   放送的には「選手集団がほぼ中心に収まり、ボールが消えても画が揺れない」を優先。
_PAN_ALPHA = 0.06          # 中心 EMA 係数（下げて追従を遅く、揺れを抑制）
_ZOOM_ALPHA = 0.04         # サイズ EMA 係数（pan より緩く＝ズーム往復を抑制）
_MAX_PAN_FRAC = 0.015      # 1 フレームの中心移動上限（フレーム幅比; 0.04→0.015 に削減）
_MAX_ZOOM_RATE = 0.02      # 1 フレームのサイズ変化上限（現サイズ比）
_DEADZONE_FRAC = 0.04      # この範囲内のターゲット中心移動は無視（0.03→0.04 に拡大）
_LOOKAHEAD_FRAMES = 1.5    # ボール速度の先読み（控えめ方向にさらに削減）
_LOOKAHEAD_MAX_FRAC = 0.08  # 先読み量の上限（フレーム幅比）

# ボール確度ごとの合成重み（加重合成の肝）。
# NG 評価でボール追従強すぎ判明 → _W_BALL_VISIBLE を 0.75→0.45 に下げる。
# 選手集団重心をアンカーにし、ボール位置は補正程度に留める（放送的挙動）。
_W_BALL_VISIBLE = 0.45     # 検出フレーム: ボールは補正役（選手集団重心が主）
_W_BALL_PREDICTED = 0.25   # Kalman 補間: さらに選手集団に寄せる
_W_BALL_LOST = 0.0         # 喪失: 選手集団のみ

# ROI サイズの制約（フレームに対する割合）。
_BALL_MARGIN = 0.30        # ボール周辺マージン（フレームサイズ比）
_WIDE_MARGIN = 0.12        # 選手 BBox 包含へのパディング（包含サイズ比）
_MIN_ROI_FRAC = 0.30       # ROI 下限（寄り過ぎ防止）
_MAX_ROI_FRAC = 1.00       # ROI 上限（フレーム全体まで）
_PAUSE_ROI_FRAC = 0.85     # ゲーム停止時に保持する wide サイズ


class FramingCalculator:
    """フレームごとの仮想カメラ ROI を算出する（滑らか平滑化つき）。

    Args:
        frame_w: 入力フレーム幅 [px]。
        frame_h: 入力フレーム高 [px]。
        output_aspect: 出力アスペクト比（w/h）。既定 16:9。
        mode: 'auto'（ボール＋選手の加重合成）／'ball'（ボール主体）／
              'wide'（選手包含主体）。'auto' が Phase 5 の本命。

    平滑化パラメータはコンストラクタ引数で上書き可能（評価・他競技調律のため）。
    """

    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        output_aspect: float = 16 / 9,
        mode: str = "auto",
        pan_alpha: float = _PAN_ALPHA,
        zoom_alpha: float = _ZOOM_ALPHA,
        max_pan_frac: float = _MAX_PAN_FRAC,
        max_zoom_rate: float = _MAX_ZOOM_RATE,
        deadzone_frac: float = _DEADZONE_FRAC,
        lookahead_frames: float = _LOOKAHEAD_FRAMES,
    ) -> None:
        self._fw = frame_w
        self._fh = frame_h
        self._aspect = output_aspect
        self._mode = mode
        self._pan_alpha = pan_alpha
        self._zoom_alpha = zoom_alpha
        self._max_pan = max_pan_frac * frame_w
        self._max_zoom_rate = max_zoom_rate
        self._deadzone = deadzone_frac * frame_w
        self._lookahead_frames = lookahead_frames

        # 平滑化状態: center [cx, cy] と size [w, h] を分離して保持。
        self._center: np.ndarray | None = None
        self._size: np.ndarray | None = None
        # ボール先読み用に直近の可視ボール位置を保持。
        self._ball_prev: np.ndarray | None = None

    # ── ターゲット（生の目標 ROI）算出 ───────────────────────────────────────────

    def _players_box(self, tracks: list[Track]) -> tuple[float, float, float, float] | None:
        """field 選手の包含 BBox（cx, cy, w, h）。選手が居なければ None。"""
        if not tracks:
            return None
        x1s = [t.bbox[0] for t in tracks]
        y1s = [t.bbox[1] for t in tracks]
        x2s = [t.bbox[2] for t in tracks]
        y2s = [t.bbox[3] for t in tracks]
        bx1, by1, bx2, by2 = min(x1s), min(y1s), max(x2s), max(y2s)
        pw = (bx2 - bx1) * _WIDE_MARGIN
        ph = (by2 - by1) * _WIDE_MARGIN
        return ((bx1 + bx2) / 2, (by1 + by2) / 2,
                (bx2 - bx1) + pw * 2, (by2 - by1) + ph * 2)

    def _ball_weight(self, ball: BallState) -> float:
        """ボール確度 → 合成重み w_ball。"""
        if not ball.visible:
            return _W_BALL_LOST
        if ball.predicted:
            return _W_BALL_PREDICTED
        return _W_BALL_VISIBLE

    def _ball_lookahead(self, ball: BallState) -> np.ndarray:
        """直近 2 フレームの可視ボール速度から先読みオフセットを返す。"""
        if not ball.visible or ball.predicted:
            self._ball_prev = None
            return np.zeros(2)
        cur = np.array([ball.x, ball.y], dtype=np.float64)
        if self._ball_prev is None:
            self._ball_prev = cur
            return np.zeros(2)
        vel = cur - self._ball_prev
        self._ball_prev = cur
        lead = vel * self._lookahead_frames
        cap = _LOOKAHEAD_MAX_FRAC * self._fw
        n = float(np.linalg.norm(lead))
        if n > cap:
            lead = lead / n * cap
        return lead

    def _target(
        self, ball: BallState, tracks: list[Track], game_active: bool
    ) -> tuple[float, float, float, float]:
        """このフレームの生ターゲット ROI（cx, cy, w, h）。平滑化前。"""
        players = self._players_box(tracks)

        # ── ゲーム停止: ボールを追わず選手集団を wide に保持 ──────────────────────
        if not game_active:
            if players is not None:
                pcx, pcy, _, _ = players
            else:
                pcx, pcy = self._fw / 2, self._fh / 2
            return pcx, pcy, self._fw * _PAUSE_ROI_FRAC, self._fh * _PAUSE_ROI_FRAC

        w_ball = self._ball_weight(ball)
        ball_xy = (np.array([ball.x, ball.y], dtype=np.float64) + self._ball_lookahead(ball)
                   if ball.visible else None)

        # ── モード別の中心とサイズ ──────────────────────────────────────────────
        if self._mode == "ball" and ball_xy is not None:
            cx, cy = ball_xy
            return cx, cy, self._fw * _BALL_MARGIN * 2, self._fh * _BALL_MARGIN * 2

        if self._mode == "wide" or (players is None and ball_xy is None):
            if players is not None:
                return players
            return self._fw / 2, self._fh / 2, self._fw * 0.8, self._fh * 0.8

        # auto: ボールと選手集団を w_ball で加重合成（ハード切替なし）。
        if players is None:
            # 選手が居ない（稀）→ ボール主体。
            cx, cy = ball_xy
            return cx, cy, self._fw * _BALL_MARGIN * 2, self._fh * _BALL_MARGIN * 2

        pcx, pcy, pw, ph = players
        if ball_xy is None:
            # ボール喪失 → 選手集団のみ。
            return pcx, pcy, pw, ph

        bcx, bcy = ball_xy
        cx = w_ball * bcx + (1.0 - w_ball) * pcx
        cy = w_ball * bcy + (1.0 - w_ball) * pcy
        # サイズもボールタイト枠と選手包含枠を加重合成。ボール主体ほど寄る。
        bw, bh = self._fw * _BALL_MARGIN * 2, self._fh * _BALL_MARGIN * 2
        w = w_ball * bw + (1.0 - w_ball) * pw
        h = w_ball * bh + (1.0 - w_ball) * ph
        return cx, cy, w, h

    # ── 平滑化（pan / zoom 分離＋速度制限＋デッドゾーン） ──────────────────────────

    def _smooth_center(self, target: np.ndarray) -> np.ndarray:
        assert self._center is not None
        delta = target - self._center
        dist = float(np.linalg.norm(delta))
        # デッドゾーン内のターゲット移動は無視（微小揺れのジッター抑制）。
        if dist < self._deadzone:
            return self._center
        step = self._pan_alpha * delta
        # 1 フレームの pan 量を上限でクランプ（急なカメラ振りを防ぐ）。
        snorm = float(np.linalg.norm(step))
        if snorm > self._max_pan:
            step = step / snorm * self._max_pan
        return self._center + step

    def _smooth_size(self, target: np.ndarray) -> np.ndarray:
        assert self._size is not None
        step = self._zoom_alpha * (target - self._size)
        # 1 フレームのサイズ変化を現サイズ比で上限クランプ（ズーム往復を抑制）。
        cap = self._max_zoom_rate * self._size
        step = np.clip(step, -cap, cap)
        return self._size + step

    def _clamp_roi(self, cx: float, cy: float, w: float, h: float) -> ROI:
        """アスペクト維持・最小/最大サイズ・フレーム内クランプを適用。

        フレーム境界クランプ後にアスペクト比を再適用する。
        フレームが出力アスペクト比より横長（例: 2160×650 ≈ 3.3:1）の場合、
        高さは常にフレーム高に頭打ちされ、幅が h×aspect で確定する。
        この再計算を省くとフレームごとにアスペクト比がばらつき映像が歪む。
        """
        w = float(np.clip(w, self._fw * _MIN_ROI_FRAC, self._fw * _MAX_ROI_FRAC))
        h = float(np.clip(h, self._fh * _MIN_ROI_FRAC, self._fh * _MAX_ROI_FRAC))
        # 目標アスペクト比に合わせる。
        if w / h > self._aspect:
            h = w / self._aspect
        else:
            w = h * self._aspect
        # フレーム境界クランプ後、はみ出した側を縮め、もう一方の辺もアスペクト再適用。
        if w > self._fw:
            w = float(self._fw)
            h = w / self._aspect
        if h > self._fh:
            h = float(self._fh)
            w = h * self._aspect
        # 高さ再クランプ後に幅がまたはみ出す可能性（超横長フレーム）。
        w = min(w, self._fw)
        x = int(np.clip(cx - w / 2, 0, self._fw - w))
        y = int(np.clip(cy - h / 2, 0, self._fh - h))
        return ROI(x=x, y=y, w=int(w), h=int(h))

    def compute(
        self,
        ball: BallState,
        tracks: list[Track],
        game_active: bool = True,
    ) -> ROI:
        """このフレームの平滑化済みクロップ ROI を返す。

        Args:
            ball:        BallState（visible / predicted / lost）。
            tracks:      framing 対象の field 選手トラック。
            game_active: participation.game_active。False（タイムアウト／セット間）
                         ではボールを追わず wide に保持する。既定 True。
        """
        tcx, tcy, tw, th = self._target(ball, tracks, game_active)
        target_c = np.array([tcx, tcy], dtype=np.float64)
        target_s = np.array([tw, th], dtype=np.float64)

        if self._center is None:
            self._center = target_c.copy()
            self._size = target_s.copy()
        else:
            self._center = self._smooth_center(target_c)
            self._size = self._smooth_size(target_s)

        return self._clamp_roi(self._center[0], self._center[1],
                               self._size[0], self._size[1])
