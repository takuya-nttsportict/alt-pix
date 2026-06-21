"""Role classification: field player / bench / referee / unknown.

Rationale (CLAUDE.md Phase 4, docs/phase4_player_perception.md):
  Framing and analytics both need to ignore people who are *on screen* but not
  *in play*: substitutes, referees, line judges, bystanders caught on camera.

  2026-06 redesign (evaluation #4): position alone cannot do this in volleyball
  — the server legally steps off court (and is the key person to keep framed),
  and the 1st/2nd referees stand at the net centre. So the primary axis is now
  TEMPORAL PARTICIPATION (see participation.py): integrated over the game, a
  player has a high on-court fraction and large movement; a line judge / bench /
  bystander does not. Position is no longer a hard gate but one of the
  accumulated features (with the serve handled implicitly by time integration).

Decision (once a track has enough history):
  participating (score >= threshold)  → "field"
  not participating + team-coloured    → "bench"   (substitute in team kit)
  not participating + colour outlier    → "referee" (non-player: ref/line/bystander)
  not participating + no colour signal  → "referee" (coarse non-player bucket)
  too few frames so far                 → "off"     (defer until evidence builds)

The bench/referee split needs a uniform-colour signal (dist_map). Without it
(e.g. court-half team assignment carries no colour), non-players land in the
coarse "referee" bucket — acceptable for framing, where both are deprioritised.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from .court import CourtCalibration
from .participation import ParticipationTracker
from .tracker import Track

logger = logging.getLogger(__name__)

Role = Literal["field", "bench", "referee", "off", "unknown"]

# A non-participant is split bench vs referee by colour: distance to the nearest
# team centroid above OUTLIER_K robust spreads of the field players' distances
# marks a colour outlier (referee). Robust (median/MAD) so a few referees do not
# inflate the threshold.
_OUTLIER_K = 3.0
_MIN_REF_SAMPLES = 6


class RoleClassifier:
    """Assign a play-role from accumulated participation + a colour outlier split.

    Stateful: owns a ParticipationTracker (per-track evidence over the game) and
    an EMA of the referee colour-outlier threshold so the boundary stays stable.
    """

    def __init__(
        self,
        court: CourtCalibration,
        field_margin: float = 60.0,
        participation: ParticipationTracker | None = None,
    ) -> None:
        self._court = court
        self._field_margin = field_margin
        self._part = participation or ParticipationTracker(court)
        self._thresh_ema: float | None = None
        self._last_reasons: dict[int, str] = {}

    @property
    def last_reasons(self) -> dict[int, str]:
        """{track_id: human-readable reason} from the most recent classify()."""
        return self._last_reasons

    @property
    def participation(self) -> ParticipationTracker:
        return self._part

    def _appear_match_from_dist(
        self, tracks: list[Track], dist_map: dict[int, float]
    ) -> dict[int, float] | None:
        """Map colour distance-to-team into a [0,1] uniform-match score.

        Uses the robust spread of all tracks' distances this frame; close to a
        team centroid → ~1 (looks like a uniform), far → ~0 (colour outlier).
        Returns None when there is no usable colour signal.
        """
        vals = [dist_map[t.track_id] for t in tracks
                if t.track_id in dist_map and np.isfinite(dist_map[t.track_id])]
        if len(vals) < _MIN_REF_SAMPLES:
            return None
        arr = np.asarray(vals, dtype=np.float64)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med))) + 1e-6
        scale = _OUTLIER_K * 1.4826 * mad
        out: dict[int, float] = {}
        for t in tracks:
            d = dist_map.get(t.track_id)
            if d is None or not np.isfinite(d):
                continue
            # match = 1 at/below median, decaying to 0 by med + scale.
            out[t.track_id] = float(np.clip(1.0 - max(0.0, d - med) / max(scale, 1e-6), 0.0, 1.0))
        return out

    def classify(
        self,
        tracks: list[Track],
        team_map: dict[int, int],
        dist_map: dict[int, float],
    ) -> dict[int, Role]:
        """Return {track_id: role}.

        Args:
            tracks:   active player tracks.
            team_map: {track_id: team 0/1 or -1} (used only for the bench split).
            dist_map: {track_id: colour distance to nearest team centroid}.
                      Empty when team assignment carries no colour (court-half).
        """
        appear_match = self._appear_match_from_dist(tracks, dist_map) if dist_map else None
        states = self._part.update(tracks, appear_match)

        # Referee colour threshold from non-participating, off-court colour dists.
        ref_dists = [dist_map[t.track_id] for t in tracks
                     if t.track_id in dist_map and np.isfinite(dist_map.get(t.track_id, np.nan))]
        thr = self._update_threshold(ref_dists)

        roles: dict[int, Role] = {}
        reasons: dict[int, str] = {}
        for t in tracks:
            tid = t.track_id
            st = states.get(tid)
            expl = self._part.explain(tid)

            # Fast-in: only players ever enter the court, so anyone on the court
            # surface RIGHT NOW is a field player — no warm-up needed (the user's
            # "judge entering as quickly as possible").
            if st is not None and st.on_court_now:
                roles[tid] = "field"
                reasons[tid] = f"on court now -> field [{expl}]"
                continue

            # Slow-out: keep "field" long after a player steps out. An end-line
            # exit is almost certainly a serve (assume_server) and gets a much
            # longer grace than a sideline ball-chase.
            if st is not None and st.recently_exited:
                roles[tid] = "field"
                tag = "assume server" if st.assume_server else "recently exited court"
                reasons[tid] = f"{tag} ({st.frames_since_exit}f ago) -> field [{expl}]"
                continue

            if st is None or st.frames < self._part.min_frames:
                roles[tid] = "off"
                reasons[tid] = f"gathering evidence -> off [{expl}]"
                continue

            if st.score >= self._part.player_threshold:
                roles[tid] = "field"
                reasons[tid] = f"participating -> field [{expl}]"
                continue

            # Non-participant: split bench (team colour) vs referee (outlier/none).
            d = dist_map.get(tid)
            if team_map.get(tid, -1) >= 0 and d is not None and np.isfinite(d) and thr is not None and d <= thr:
                roles[tid] = "bench"
                reasons[tid] = f"not participating + team colour (d={d:.3f}<=thr={thr:.3f}) -> bench [{expl}]"
            else:
                roles[tid] = "referee"
                why = (f"colour outlier d={d:.3f}>thr={thr:.3f}"
                       if (d is not None and np.isfinite(d) and thr is not None)
                       else "no colour signal")
                reasons[tid] = f"not participating + {why} -> referee [{expl}]"

        self._last_reasons = reasons
        return roles

    def _update_threshold(self, dists: list[float]) -> float | None:
        if len(dists) < _MIN_REF_SAMPLES:
            return self._thresh_ema
        arr = np.asarray(dists, dtype=np.float64)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med))) + 1e-6
        thr = med + _OUTLIER_K * 1.4826 * mad
        self._thresh_ema = thr if self._thresh_ema is None else 0.9 * self._thresh_ema + 0.1 * thr
        return self._thresh_ema
