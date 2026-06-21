"""Game-state estimation for state-aware framing (Phase 5).

放送カメラワークの先行研究（Ariki 2006 / Pixellot / volleyball_analytics）は一致して
「カメラ挙動はゲーム状態で切り替えるべき」と示す。本モジュールはバレーボールの局面を
3 状態に粗く推定し、フレーミング（framing.py）がズームレベル・パン挙動を変えるための
シグナルを供給する。

  RALLY    : ラリー進行中。ボールが空中〜コート上で active。
             → 通常追従（選手集団アンカー＋ボール補正、中ズーム）。
  SERVICE  : サーブ局面。ボールがエンドライン付近／後方にあり、サーバーが
             コート外（endline ゾーン）から打つ。
             → サーバー＋レシーブ隊形が収まる wide にゆっくり引く。
  NO_PLAY  : デッドボール・タイムアウト・セット間。コート上人数 EMA が低い、
             またはボール喪失が続く。
             → パンを凍結し wide で保持（選手の出入りを追わない）。

注意（プリンシパル4・8）:
  SERVICE / NO_PLAY の判定は現時点では**ヒューリスティック**である。確実な
  サーブ検出・デッドボール検出には action recognition（Phase 6）が要るため、
  ここでは利用可能なシグナル（ball 可視性・コート射影・participation の
  on-court 人数 EMA）から最善推定し、ヒステリシスで状態のばたつきを抑える。
  将来 action recognition が入ったら本推定器を差し替える（framing 側は状態
  enum にのみ依存するので影響を受けない）。
"""

from __future__ import annotations

import logging
from typing import Literal

from .ball_tracker import BallState
from .court import CourtCalibration, _COURT_W
from .participation import ParticipationTracker
from .tracker import Track

logger = logging.getLogger(__name__)

GameState = Literal["rally", "service", "no_play"]

# サーブ判定: ボールの足元射影 u がエンドラインからこの距離 [m] 以内／外側なら
# サーブ局面の可能性（コート長 _COURT_W=18m, エンドラインは u=0 と u=18）。
_SERVE_ENDLINE_M = 2.0
# 状態切替のヒステリシス: 新状態がこのフレーム数連続で支持されて初めて遷移する
# （1 フレームのノイズでズームが暴れないように）。
_SWITCH_HYSTERESIS = 8


class GameStateEstimator:
    """ボール・選手・participation から RALLY / SERVICE / NO_PLAY を推定する。

    ヒステリシス付き。`court` が無い場合はサーブの空間判定ができないため
    RALLY / NO_PLAY の 2 状態に縮退する。
    """

    def __init__(
        self,
        court: CourtCalibration | None,
        participation: ParticipationTracker | None,
        serve_endline_m: float = _SERVE_ENDLINE_M,
        hysteresis: int = _SWITCH_HYSTERESIS,
    ) -> None:
        self._court = court
        self._part = participation
        self._serve_m = serve_endline_m
        self._hysteresis = hysteresis
        self._state: GameState = "rally"
        self._pending: GameState | None = None
        self._pending_count = 0
        self._last_reason = "init"

    @property
    def state(self) -> GameState:
        return self._state

    @property
    def last_reason(self) -> str:
        return self._last_reason

    def _instant(self, ball: BallState, tracks: list[Track]) -> tuple[GameState, str]:
        """ヒステリシス前の、このフレーム単独の生判定。"""
        # NO_PLAY: participation がコート上人数の激減を検出（タイムアウト/セット間）。
        if self._part is not None and not self._part.game_active:
            n = self._part.on_court_count_ema
            return "no_play", f"on-court ema={n:.1f} low (timeout/break)"

        # ボールが見えない（喪失が続く）→ デッドボール寄り。ただし participation が
        # active を保っているうちは RALLY を維持（瞬断はラリー継続）。framing 側の
        # 重み付けが喪失を吸収するので、ここでは NO_PLAY に倒さない。
        if not ball.visible:
            return "rally", "ball lost but game active -> keep rally"

        # SERVICE: ボール足元射影がエンドライン付近／後方。court 必須。
        if self._court is not None:
            try:
                u, _v = self._court.image_to_court(ball.x, ball.y)
            except Exception:
                u = None
            if u is not None:
                near_end = u < self._serve_m or u > (_COURT_W - self._serve_m)
                if near_end:
                    side = "near u=0" if u < self._serve_m else "near u=18"
                    return "service", f"ball at endline ({side}, u={u:.1f}m)"

        return "rally", "ball in play"

    def update(self, ball: BallState, tracks: list[Track]) -> GameState:
        """1 フレーム更新し、ヒステリシス適用後の状態を返す。"""
        inst, reason = self._instant(ball, tracks)

        if inst == self._state:
            self._pending = None
            self._pending_count = 0
            self._last_reason = reason
            return self._state

        # 状態が変わろうとしている: 同じ候補が連続 hysteresis 回支持されたら遷移。
        # NO_PLAY への遷移は participation 由来で既に EMA 平滑なので即時許可する。
        threshold = 1 if inst == "no_play" else self._hysteresis
        if inst == self._pending:
            self._pending_count += 1
        else:
            self._pending = inst
            self._pending_count = 1

        if self._pending_count >= threshold:
            logger.debug("game state %s -> %s (%s)", self._state, inst, reason)
            self._state = inst
            self._pending = None
            self._pending_count = 0
        self._last_reason = reason
        return self._state
