"""チーム割当戦略（競技種別で差し替え可能）。

なぜ「戦略」を分けるか（プリンシパル4）:
  チーム分けの最良の手掛かりは競技で異なる。
    - バレー/卓球/テニス等のネット競技: 両チームはネットで物理的に分離され、
      ラリー中に相手コートへ侵入しない。→ **コートのどちら側か**が決定的で、
      色に依存せずリベロ（規則で別色）も同じ側として正しく束ねられる。
    - バスケ/サッカー等の侵入型競技: 両チームが全面を入り混じって動く。
      → 空間では割れず、**ユニフォーム色**のクラスタリングが妥当。
  競技種別は入力で与えられる前提（--sport）。種別に応じて戦略を選ぶ。

共通インターフェース `TeamAssigner`:
    update(frame, tracks) -> TeamResult
  TeamResult は team_map / reason_map（判定根拠の説明）に加え、役割フィルタの
  審判外れ値検出に使う dist_map / margin_map を返す（無い戦略では空）。

説明可能性（プリンシパル7）:
  各 track に「なぜそのチームか」を人間可読な reason として付ける。
  例: court-half → "court u=4.2m <9m(net) → side L"
      color       → "color d0=0.12 d1=0.55 → team0 (margin 0.64)"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from .court import CourtCalibration, _COURT_W
from .team_classifier import TeamClassifier
from .tracker import Track

logger = logging.getLogger(__name__)

Sport = Literal["volleyball", "basketball", "generic"]

# ネット競技ではコート長辺の中央（_COURT_W/2 [m]）がネット。足元の court 座標 u が
# これ未満なら片側（team 0 = "L"）、以上なら反対側（team 1 = "R"）。
_NET_U = _COURT_W / 2.0


@dataclass
class TeamResult:
    """1 フレーム分のチーム割当結果。"""

    team_map: dict[int, int] = field(default_factory=dict)        # tid -> 0/1/-1
    reason_map: dict[int, str] = field(default_factory=dict)      # tid -> 説明
    dist_map: dict[int, float] = field(default_factory=dict)      # tid -> 色距離（審判検出用）
    margin_map: dict[int, float] = field(default_factory=dict)    # tid -> 確信度


class TeamAssigner(Protocol):
    @property
    def ready(self) -> bool: ...
    @property
    def method(self) -> str: ...
    def update(self, frame: np.ndarray, tracks: list[Track]) -> TeamResult: ...


# ── ネット競技: コート左右ハーフで割当 ─────────────────────────────────────────────

class CourtHalfAssigner:
    """ネット競技用。足元を court 座標へ射影し、ネット中央線の左右でチーム決定。

    決定的（warm-up 不要）・リベロ耐性あり・追加依存ゼロ。ネット境界はラリー中の
    物理的ハード境界なので、色が似ていても破綻しない。

    margin はネットからの距離 [m] を正規化した確信度（ネット際ほど低い）。
    dist_map は色を使わないので空（審判の色外れ値検出は #4 で別途）。
    """

    def __init__(self, court: CourtCalibration, ambiguous_band_m: float = 0.5) -> None:
        self._court = court
        self._band = ambiguous_band_m  # ネット ±この幅 [m] は確信度を下げる

    @property
    def ready(self) -> bool:
        return True  # 幾何のみ。常に判定可能。

    @property
    def method(self) -> str:
        return "court-half(net)"

    def update(self, frame: np.ndarray, tracks: list[Track]) -> TeamResult:
        res = TeamResult()
        for t in tracks:
            x1, y1, x2, y2 = t.bbox
            foot_x, foot_y = (x1 + x2) / 2.0, y2
            u, v = self._court.image_to_court(foot_x, foot_y)
            team = 0 if u < _NET_U else 1
            side = "L" if team == 0 else "R"
            res.team_map[t.track_id] = team
            res.reason_map[t.track_id] = (
                f"court u={u:.1f}m {'<' if team == 0 else '>='}{_NET_U:.0f}m(net) -> side {side}"
            )
            # ネットからの距離 [m] を [0,1] に潰した確信度（band 内は線形に低下）。
            dist_m = abs(u - _NET_U)
            res.margin_map[t.track_id] = float(min(1.0, dist_m / max(self._band, 1e-6)))
        return res


# ── 侵入型競技 / 汎用: ユニフォーム色クラスタで割当 ──────────────────────────────────

class ColorClusterAssigner:
    """侵入型競技・汎用。既存 TeamClassifier（SigLIP/Lab → 2-means）をラップ。

    両チームが全面を動く競技では空間で割れないため、ユニフォーム色で自己校正する。
    色距離（dist_map）と確信度（margin_map）を返すので、役割フィルタの審判外れ値
    検出にそのまま使える。
    """

    def __init__(self, classifier: TeamClassifier) -> None:
        self._clf = classifier

    @property
    def ready(self) -> bool:
        return self._clf.ready

    @property
    def method(self) -> str:
        return f"color-cluster({self._clf.backend})"

    def update(self, frame: np.ndarray, tracks: list[Track]) -> TeamResult:
        team_map, dist_map = self._clf.update(frame, tracks)
        margin_map = self._clf.last_margins
        res = TeamResult(team_map=dict(team_map), dist_map=dict(dist_map),
                         margin_map=dict(margin_map))
        for tid, team in team_map.items():
            if team < 0:
                res.reason_map[tid] = "color: warming up -> unassigned"
            else:
                d = dist_map.get(tid, float("nan"))
                m = margin_map.get(tid, float("nan"))
                res.reason_map[tid] = (
                    f"color d={d:.3f} margin={m:.2f} -> team{team}"
                )
        return res


# ── ファクトリ ───────────────────────────────────────────────────────────────────

def make_team_assigner(
    sport: Sport,
    court: CourtCalibration | None,
    classifier: TeamClassifier | None = None,
) -> TeamAssigner:
    """競技種別から戦略を選ぶ。

    volleyball（ネット競技）: コート左右ハーフ（court 必須）。
    basketball / generic     : 色クラスタ（classifier 必須）。

    バレーでも court が無い場合は色クラスタへフォールバック。
    """
    if sport == "volleyball":
        if court is None:
            logger.warning(
                "CourtHalfAssigner requires court calibration; "
                "falling back to colour clustering."
            )
        else:
            logger.info("TeamAssigner: court-half (net boundary) [volleyball]")
            return CourtHalfAssigner(court)

    if classifier is None:
        raise ValueError(
            f"sport={sport!r} needs colour clustering but no TeamClassifier given"
        )
    logger.info(f"TeamAssigner: colour clustering [{sport}]")
    return ColorClusterAssigner(classifier)
