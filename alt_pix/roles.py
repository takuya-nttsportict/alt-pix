"""Role classification: field player / bench / referee / unknown.

Rationale (CLAUDE.md Phase 4, docs/phase4_player_perception.md):
  - Framing and analytics both need to ignore people who are *on screen* but not
    *in play*: substitutes on the bench, referees, line judges, ball kids.
  - There is no commercial turnkey model that labels these roles, and OCR/ReID
    cannot solve it either. The universal signal is geometric + chromatic:

      (c) court geometry  → on-court vs off-court  (primary axis)
      (b) colour outlier  → referee vs bench        (auxiliary axis)

  This is robust to camera/venue changes: recalibrating the 4 court corners is
  enough, and the colour outlier is relative to the two team clusters actually
  present (no per-venue training).

Decision table:
  on-court                         → "field"
  off-court + team-coloured        → "bench"   (substitute in team kit)
  off-court + colour outlier       → "referee" (neither team's uniform)
  off-court + team clusters unknown → "off"     (warm-up; defer the split)

The referee/bench split needs the TeamClassifier centroids, so during the
team-classifier warm-up we emit the coarse "off" label rather than guess.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from .court import CourtCalibration
from .tracker import Track

logger = logging.getLogger(__name__)

Role = Literal["field", "bench", "referee", "off", "unknown"]

# A person is flagged a colour outlier (referee) when their distance to the
# nearest team centroid exceeds OUTLIER_K times the robust spread of the
# on-court players' distances. Robust (median/MAD) so a couple of referees do
# not inflate the threshold.
_OUTLIER_K = 3.0
_MIN_REF_SAMPLES = 6  # need enough field players to estimate the spread


class RoleClassifier:
    """Assign a play-role to each track from court geometry + team-colour outliers.

    Stateless per call except for an EMA of the outlier threshold, which keeps
    the referee/bench boundary stable across frames instead of flickering.
    """

    def __init__(self, court: CourtCalibration, field_margin: float = 60.0) -> None:
        self._court = court
        self._field_margin = field_margin
        self._thresh_ema: float | None = None

    def _update_threshold(self, field_dists: list[float]) -> float | None:
        """Robust outlier threshold from the colour-distance of on-court players.

        On-court players are (almost) all in team kit, so their distance-to-team
        distribution defines "normal"; a referee sits well above it.
        """
        if len(field_dists) < _MIN_REF_SAMPLES:
            return self._thresh_ema  # not enough signal; reuse last estimate
        arr = np.asarray(field_dists, dtype=np.float64)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med))) + 1e-6
        thr = med + _OUTLIER_K * 1.4826 * mad  # 1.4826 → MAD≈σ for normal data
        # EMA-smooth so the boundary does not jump frame to frame.
        self._thresh_ema = thr if self._thresh_ema is None else 0.9 * self._thresh_ema + 0.1 * thr
        return self._thresh_ema

    def classify(
        self,
        tracks: list[Track],
        team_map: dict[int, int],
        dist_map: dict[int, float],
    ) -> dict[int, Role]:
        """Return {track_id: role}.

        Args:
            tracks:   active player tracks.
            team_map: {track_id: team 0/1 or -1} from TeamClassifier.
            dist_map: {track_id: colour distance to nearest team centroid}.
                      Empty during team-classifier warm-up.
        """
        on_court = {t.track_id: self._court.is_on_court(t.bbox, self._field_margin)
                    for t in tracks}

        # Estimate the referee threshold from on-court players' colour distances.
        field_dists = [dist_map[t.track_id] for t in tracks
                       if on_court[t.track_id] and t.track_id in dist_map
                       and np.isfinite(dist_map[t.track_id])]
        thr = self._update_threshold(field_dists)

        roles: dict[int, Role] = {}
        for t in tracks:
            tid = t.track_id
            if on_court[tid]:
                roles[tid] = "field"
                continue
            # Off-court: split bench vs referee by colour outlier when possible.
            d = dist_map.get(tid)
            if team_map.get(tid, -1) < 0 or d is None or not np.isfinite(d) or thr is None:
                roles[tid] = "off"
            elif d > thr:
                roles[tid] = "referee"
            else:
                roles[tid] = "bench"
        return roles
