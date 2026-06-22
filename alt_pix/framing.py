"""Virtual camera framing: state-aware ROI with critically-damped smoothing.

Phase 5（再設計, 2026-06）— 放送カメラワークのセオリー調査を反映。
参考: Ariki 2006（状況認識でカメラワーク切替）, OHMM（軌跡平滑化）,
Unity SmoothDamp / Game Programming Gems 4（臨界制動スプリング）, 放送現場の
リードルーム／速度制限の慣行。

旧実装（速度制限つき EMA）の致命的問題:
  EMA は 1 次フィルタで「速度」を状態に持たない。目標が動き続けると EMA は
  遅れて追従し、目標が止まると慣性でオーバーシュート。さらに速度上限クランプに
  当たるとカクつく。実映像評価で「カクカクして左右に揺れる」と NG。

本実装の核心:

  1. **臨界制動スプリング（SmoothDamp）で平滑化**:
     カメラ中心（pan）とサイズ（zoom）を、速度を状態に持つ 2 次フィルタで追従。
     オーバーシュートせず最短で収束し、ease-in/ease-out が自然。1 パラメータ
     `smooth_time`（秒）で「気持ちよい追従速度」を決める。pan より zoom を
     ゆっくりにしてズームのハンチングを抑える。

  2. **ゲーム状態でカメラ挙動を切替**（game_state.py が供給）:
       RALLY    → 選手集団アンカー＋ボール補正の中ズーム追従。
       SERVICE  → サーバー＋レシーブ隊形が入る wide にゆっくり引く。
       NO_PLAY  → パンを凍結し wide で保持（出入りを追わない）。

  3. **リードルーム**: ボール進行方向（主に水平）へ中心を少しオフセット。
     放送の定石「動く先に空間を空ける」。バレーは pan 優先なので水平主体。

  4. **ズームイン演出のフック（プレースホルダ）**: スパイク/決定機での寄りは
     action recognition（Phase 6）が要る。現時点では `highlight` 引数で
     寄り→保持→戻しのエンベロープだけ実装し、トリガ信号が来たら効くようにする。

出力は ROI 座標のみ（クロップ描画は 5.x）。アスペクト比 16:9 は厳密維持
（横長素材でフレームごとに歪まないよう、境界クランプ後に必ず再適用）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .ball_tracker import BallState
from .game_state import GameState
from .tracker import Track


@dataclass
class ROI:
    x: int  # top-left column
    y: int  # top-left row
    w: int  # width
    h: int  # height


# ── 平滑化（臨界制動スプリング）の時定数 [秒] ─────────────────────────────────────
# smooth_time が大きいほど滑らか（遅い）。放送的: pan はやや機敏、zoom は緩慢。
_PAN_SMOOTH_TIME = 0.9      # 中心追従の時定数
_ZOOM_SMOOTH_TIME = 1.6     # サイズ追従の時定数（pan より緩く）
_MAX_PAN_SPEED_FRAC = 0.6   # pan の最大速度（フレーム幅/秒）。急な飛びを抑える上限

# ── リードルーム ────────────────────────────────────────────────────────────────
_LEAD_GAIN = 0.18           # ボール速度 [px/frame] に対する先行量の係数
_LEAD_MAX_FRAC = 0.12       # 先行オフセットの上限（フレーム幅比）

# ── ボール確度ごとの合成重み（RALLY）─────────────────────────────────────────────
# 選手集団重心をアンカーにし、ボールは補正役（左右スウェー回避）。
_W_BALL_VISIBLE = 0.45
_W_BALL_PREDICTED = 0.25
_W_BALL_LOST = 0.0

# ── ROI サイズの制約（フレーム高に対する割合の zoom レベル）──────────────────────
# 「zoom レベル」は ROI 高 / フレーム高（1.0=フル）。状態ごとの目標 zoom を定義。
_ZOOM_RALLY = 0.72          # ラリー通常: 中ズーム（全コートが概ね見える）
_ZOOM_SERVICE = 0.58        # サーブ: サーバーに寄せる中ズーム（ルーズな wide を避ける）
_ZOOM_SERVICE_NOFOCUS = 0.92  # サーバー位置不明時のフォールバック（従来の wide）
_ZOOM_NO_PLAY = 0.98        # デッドボール: ほぼフル wide
_ZOOM_HIGHLIGHT = 0.50      # ズームイン演出時（スパイク等、Phase 6 で発火）
_MIN_ZOOM = 0.35            # 寄りすぎ下限
_MAX_ZOOM = 1.00

_WIDE_MARGIN = 0.12         # 選手包含 BBox へのパディング（包含サイズ比）
# ハイライト・エンベロープの増減速度（1 フレームあたりの zoom 係数変化）。
_HIGHLIGHT_RATE = 0.04


def _smooth_damp(
    current: np.ndarray, target: np.ndarray, vel: np.ndarray,
    smooth_time: float, dt: float, max_speed: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """臨界制動スプリング（Unity SmoothDamp 準拠）。ベクトル対応。

    速度 `vel` を状態として持ち、オーバーシュートせず target へ収束する。
    EMA と違い目標が止まれば速度も減衰して滑らかに停止する。
    Returns (new_position, new_velocity)。
    """
    omega = 2.0 / max(smooth_time, 1e-4)
    x = omega * dt
    exp = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)
    change = current - target
    orig_target = target.copy()
    # 最大速度クランプ（ベクトルの大きさで制限）。
    if max_speed is not None:
        max_change = max_speed * smooth_time
        mag = float(np.linalg.norm(change))
        if mag > max_change and mag > 1e-9:
            change = change / mag * max_change
    shifted_target = current - change
    temp = (vel + omega * change) * dt
    new_vel = (vel - omega * temp) * exp
    out = shifted_target + (change + temp) * exp
    # オーバーシュート防止（符号反転したら target にスナップ）。
    over = (orig_target - current) * (out - orig_target) > 0
    out = np.where(over, orig_target, out)
    new_vel = np.where(over, (out - orig_target) / dt, new_vel)
    return out, new_vel


class FramingCalculator:
    """状態認識つき仮想カメラ ROI を算出する（臨界制動スプリング平滑化）。

    Args:
        frame_w, frame_h: 入力フレーム寸法 [px]。
        output_aspect: 出力アスペクト比（w/h）。既定 16:9。
        fps: 入力フレームレート（スプリングの dt 計算に使用）。
        pan_smooth_time / zoom_smooth_time: 平滑化時定数 [秒]（大きいほど滑らか）。
    """

    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        output_aspect: float = 16 / 9,
        mode: str = "auto",
        fps: float = 30.0,
        pan_smooth_time: float = _PAN_SMOOTH_TIME,
        zoom_smooth_time: float = _ZOOM_SMOOTH_TIME,
        max_pan_speed_frac: float = _MAX_PAN_SPEED_FRAC,
    ) -> None:
        self._fw = frame_w
        self._fh = frame_h
        self._aspect = output_aspect
        self._mode = mode  # 'auto'（状態認識）/'ball'/'wide'（手動上書き）
        self._dt = 1.0 / max(fps, 1e-3)
        self._pan_t = pan_smooth_time
        self._zoom_t = zoom_smooth_time
        self._max_pan_speed = max_pan_speed_frac * frame_w

        # スプリング状態（位置＋速度）。
        self._center: np.ndarray | None = None
        self._center_vel = np.zeros(2)
        self._size: np.ndarray | None = None
        self._size_vel = np.zeros(2)
        # リードルーム用の直近可視ボール位置。
        self._ball_prev: np.ndarray | None = None
        # ハイライト（ズームイン演出）エンベロープ 0..1。
        self._highlight_env = 0.0

    # ── 部品 ───────────────────────────────────────────────────────────────────

    def _players_box(self, tracks: list[Track]) -> tuple[float, float, float, float] | None:
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
        if not ball.visible:
            return _W_BALL_LOST
        return _W_BALL_PREDICTED if ball.predicted else _W_BALL_VISIBLE

    def _lead_room(self, ball: BallState) -> np.ndarray:
        """ボール進行方向（水平主体）への先行オフセット。"""
        if not ball.visible or ball.predicted:
            self._ball_prev = None
            return np.zeros(2)
        cur = np.array([ball.x, ball.y], dtype=np.float64)
        if self._ball_prev is None:
            self._ball_prev = cur
            return np.zeros(2)
        vel = cur - self._ball_prev
        self._ball_prev = cur
        # バレーは pan 優先 → 水平を主に、垂直は控えめ（×0.3）。
        lead = np.array([vel[0], vel[1] * 0.3]) * _LEAD_GAIN
        cap = _LEAD_MAX_FRAC * self._fw
        n = float(np.linalg.norm(lead))
        if n > cap:
            lead = lead / n * cap
        return lead

    def _zoom_level(self, game_state: GameState, highlight: bool, has_focus: bool) -> float:
        """状態 → 目標 zoom レベル（ROI 高 / フレーム高）。ハイライトで寄る。"""
        service_zoom = _ZOOM_SERVICE if has_focus else _ZOOM_SERVICE_NOFOCUS
        base = {
            "rally": _ZOOM_RALLY,
            "service": service_zoom,
            "no_play": _ZOOM_NO_PLAY,
        }.get(game_state, _ZOOM_RALLY)
        # ハイライト・エンベロープを更新（寄り→保持→戻しの滑らかな係数）。
        tgt = 1.0 if highlight else 0.0
        if self._highlight_env < tgt:
            self._highlight_env = min(tgt, self._highlight_env + _HIGHLIGHT_RATE)
        else:
            self._highlight_env = max(tgt, self._highlight_env - _HIGHLIGHT_RATE)
        # エンベロープで base と highlight zoom を補間。
        z = base + (_ZOOM_HIGHLIGHT - base) * self._highlight_env
        return float(np.clip(z, _MIN_ZOOM, _MAX_ZOOM))

    def _target(
        self, ball: BallState, tracks: list[Track], game_state: GameState,
        highlight: bool, focus_xy: tuple[float, float] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """生ターゲット（center[2], size[2]）。スプリング前。"""
        players = self._players_box(tracks)
        has_focus = game_state == "service" and focus_xy is not None
        zoom = self._zoom_level(game_state, highlight, has_focus)
        size_h = zoom * self._fh
        size_w = size_h * self._aspect
        size = np.array([size_w, size_h], dtype=np.float64)

        # 中心は手動モードを尊重しつつ、状態で重み付け。
        if self._mode == "wide":
            if players is not None:
                cx, cy, _, _ = players
            else:
                cx, cy = self._fw / 2, self._fh / 2
            return np.array([cx, cy]), size

        # SERVICE: サーバー（assume_server）をフレームの「コート側の端」に置く。
        # 中央に置くとサーバー背後の壁が半分を占めて無駄になる。
        # サーバーが画面左半分 → サーバーをフレーム左端付近に配置（コートが右に広がる）。
        # サーバーが画面右半分 → サーバーをフレーム右端付近に配置（コートが左に広がる）。
        if has_focus and focus_xy is not None:
            sx = float(focus_xy[0])
            sy = float(focus_xy[1])
            _EDGE_MARGIN = 0.08   # サーバーをフレーム端から何割の余白に置くか
            half_w = size[0] / 2.0
            if sx < self._fw / 2:
                # 左サーバー: ROI 中心をサーバーの右側へ（サーバーを左端に）
                cx = sx + half_w * (1.0 - _EDGE_MARGIN * 2)
            else:
                # 右サーバー: ROI 中心をサーバーの左側へ（サーバーを右端に）
                cx = sx - half_w * (1.0 - _EDGE_MARGIN * 2)
            return np.array([cx, sy], dtype=np.float64), size

        if game_state == "no_play":
            # パン凍結: 既存中心を維持（出入りを追わない）。初期は選手 or 画面中央。
            if self._center is not None:
                return self._center.copy(), size
            if players is not None:
                return np.array([players[0], players[1]]), size
            return np.array([self._fw / 2, self._fh / 2]), size

        pcx, pcy = (players[0], players[1]) if players else (self._fw / 2, self._fh / 2)
        w_ball = self._ball_weight(ball)
        if self._mode == "ball" and ball.visible:
            w_ball = 1.0

        if ball.visible and w_ball > 0:
            bxy = np.array([ball.x, ball.y], dtype=np.float64) + self._lead_room(ball)
            cx = w_ball * bxy[0] + (1 - w_ball) * pcx
            cy = w_ball * bxy[1] + (1 - w_ball) * pcy
        else:
            self._lead_room(ball)  # ボール履歴をリセット
            cx, cy = pcx, pcy
        return np.array([cx, cy], dtype=np.float64), size

    # ── 仕上げ（アスペクト維持・クランプ）────────────────────────────────────────

    def _clamp_roi(self, cx: float, cy: float, w: float, h: float) -> ROI:
        """アスペクト維持・最小/最大・フレーム内クランプ。境界後にアスペクト再適用。"""
        w = float(np.clip(w, self._fw * _MIN_ZOOM * self._aspect, self._fw))
        h = float(np.clip(h, self._fh * _MIN_ZOOM, self._fh))
        if w / h > self._aspect:
            h = w / self._aspect
        else:
            w = h * self._aspect
        if w > self._fw:
            w = float(self._fw)
            h = w / self._aspect
        if h > self._fh:
            h = float(self._fh)
            w = h * self._aspect
        w = min(w, self._fw)
        x = int(np.clip(cx - w / 2, 0, self._fw - w))
        y = int(np.clip(cy - h / 2, 0, self._fh - h))
        return ROI(x=x, y=y, w=int(w), h=int(h))

    def compute(
        self,
        ball: BallState,
        tracks: list[Track],
        game_state: GameState = "rally",
        highlight: bool = False,
        focus_xy: tuple[float, float] | None = None,
    ) -> ROI:
        """このフレームの平滑化済み ROI を返す。

        Args:
            ball:       BallState（visible / predicted / lost）。
            tracks:     framing 対象の field 選手トラック。
            game_state: "rally" / "service" / "no_play"（game_state.py が供給）。
            highlight:  ズームイン演出のトリガ（スパイク等; 現状プレースホルダ）。
            focus_xy:   SERVICE 時に優先して寄せる点（サーバー位置 [px]）。
                        与えられれば SERVICE はルーズな wide でなくサーバー中ズーム。
        """
        target_c, target_s = self._target(ball, tracks, game_state, highlight, focus_xy)

        if self._center is None:
            self._center = target_c.copy()
            self._size = target_s.copy()
        else:
            self._center, self._center_vel = _smooth_damp(
                self._center, target_c, self._center_vel,
                self._pan_t, self._dt, self._max_pan_speed)
            self._size, self._size_vel = _smooth_damp(
                self._size, target_s, self._size_vel, self._zoom_t, self._dt)

        return self._clamp_roi(self._center[0], self._center[1],
                               self._size[0], self._size[1])
