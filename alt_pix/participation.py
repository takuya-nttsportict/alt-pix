"""Per-track participation estimation: is this person actually playing?

Why this exists (Phase 4 evaluation #4, redesigned 2026-06):
  Volleyball referees cannot be separated from players by *position* alone:
    - the server legally steps off the court (behind the end line) — and the
      server is the single most important person to keep framed (Phase 3.5),
    - the 1st/2nd referees stand at the net CENTRE, not in a corner,
  so an on-court / off-court binary mislabels both.

  The robust, sport-agnostic signal is TEMPORAL: integrated over a whole game,
  a player and a non-player (line judge, bench, a bystander caught on camera)
  diverge clearly, regardless of any single frame:

    feature (accumulated)      player        non-player (line judge / bystander)
    --------------------------  ------------  -----------------------------------
    on-court time fraction      high          ~0   (always outside)
    total movement              large         small (nearly stationary)
    uniform-colour match        matches team  outlier (optional 3rd signal)

  The server is handled *for free*: they are inside the court the vast majority
  of the match and only briefly outside during their own serve, so their
  on-court fraction stays high — no explicit serve detector needed.

The tracker keeps a slow EMA per track so the estimate only sharpens as the
match runs (the user's intent: accumulate evidence over the game). Early in a
track's life the estimate is withheld (`unknown`) until enough frames are seen.

Pure numpy; unit-testable without torch/onnxruntime.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .court import CourtCalibration
from .tracker import Track

logger = logging.getLogger(__name__)


@dataclass
class _PartState:
    frames: int = 0
    on_court_ema: float = 0.0       # EMA of on-court indicator [0,1]
    motion_ema: float = 0.0         # EMA of foot displacement / bbox-height
    appear_ema: float | None = None  # EMA of uniform-colour match [0,1] (optional)
    last_foot: tuple[float, float] | None = None
    score: float = 0.0
    # Grace period: track when person was last on court.
    frames_since_exit: int = 999    # frames elapsed since leaving the court (999 = never inside)
    was_on_court: bool = False      # on-court last frame
    recently_exited: bool = False   # True within grace_frames after leaving court


@dataclass
class ParticipationTracker:
    """Accumulate per-track evidence that a person is (not) playing.

    Args:
        court:            calibration to test on-court membership.
        on_court_margin:  px margin around the court polygon counted as "in".
        alpha:            EMA weight for new evidence (small = long memory).
        motion_ref:       per-frame foot displacement (in bbox-height units) that
                          counts as "clearly moving"; motion_ema is scaled by it.
        min_frames:       a track must be observed this many frames before its
                          participation is judged (else `unknown`).
        player_threshold: score >= this → playing (field).
        w_on_court / w_motion / w_appearance:
                          feature weights. Appearance is dropped (weights
                          renormalised) for any track with no colour signal.
    """

    court: CourtCalibration
    on_court_margin: float = 0.0
    alpha: float = 0.02
    motion_ref: float = 0.04
    min_frames: int = 90
    player_threshold: float = 0.5
    w_on_court: float = 0.4
    w_motion: float = 0.3
    w_appearance: float = 0.3
    grace_frames: int = 45  # frames after leaving court to keep recently_exited=True (~1.5s@30fps)

    _states: dict[int, _PartState] = field(default_factory=dict)

    def update(
        self,
        tracks: list[Track],
        appear_match: dict[int, float] | None = None,
    ) -> dict[int, _PartState]:
        """Fold this frame's evidence into each track's running estimate.

        Args:
            tracks:       active tracks this frame.
            appear_match: optional {track_id: uniform-colour match in [0,1]}
                          (1 = looks like a team uniform, 0 = colour outlier).
                          When absent, only geometry + motion are used.

        Returns the (updated) state for every active track.
        """
        out: dict[int, _PartState] = {}
        for t in tracks:
            st = self._states.setdefault(t.track_id, _PartState())
            st.frames += 1

            # ── on-court indicator ────────────────────────────────────────────
            on_court_now = self.court.is_on_court(t.bbox, self.on_court_margin)
            on = 1.0 if on_court_now else 0.0
            st.on_court_ema += self.alpha * (on - st.on_court_ema)

            # ── grace period: recently exited court ───────────────────────────
            if on_court_now:
                st.frames_since_exit = 0
                st.was_on_court = True
            elif st.was_on_court:
                # just exited or still off after having been on
                st.frames_since_exit += 1
            st.recently_exited = (st.frames_since_exit > 0
                                  and st.frames_since_exit <= self.grace_frames)

            # ── motion (bbox-height-normalised foot displacement) ─────────────
            x1, y1, x2, y2 = t.bbox
            foot = ((x1 + x2) / 2.0, float(y2))
            bh = max(y2 - y1, 1.0)
            if st.last_foot is not None:
                disp = float(np.hypot(foot[0] - st.last_foot[0],
                                      foot[1] - st.last_foot[1])) / bh
            else:
                disp = 0.0
            st.last_foot = foot
            st.motion_ema += self.alpha * (disp - st.motion_ema)

            # ── appearance (optional) ─────────────────────────────────────────
            if appear_match is not None and t.track_id in appear_match:
                a = float(appear_match[t.track_id])
                st.appear_ema = a if st.appear_ema is None else (
                    st.appear_ema + self.alpha * (a - st.appear_ema)
                )

            st.score = self._score(st)
            out[t.track_id] = st
        return out

    def _score(self, st: _PartState) -> float:
        motion_score = min(1.0, st.motion_ema / max(self.motion_ref, 1e-6))
        comps = [st.on_court_ema, motion_score]
        weights = [self.w_on_court, self.w_motion]
        if st.appear_ema is not None:
            comps.append(st.appear_ema)
            weights.append(self.w_appearance)
        wsum = sum(weights)
        return float(sum(c * w for c, w in zip(comps, weights)) / wsum) if wsum else 0.0

    def is_ready(self, track_id: int) -> bool:
        st = self._states.get(track_id)
        return st is not None and st.frames >= self.min_frames

    def explain(self, track_id: int) -> str:
        """Human-readable breakdown of a track's participation estimate."""
        st = self._states.get(track_id)
        if st is None:
            return "no data"
        motion_score = min(1.0, st.motion_ema / max(self.motion_ref, 1e-6))
        ap = f" appear={st.appear_ema:.2f}" if st.appear_ema is not None else ""
        grace = f" grace={st.frames_since_exit}f" if st.recently_exited else ""
        return (f"score={st.score:.2f} (oncourt={st.on_court_ema:.2f} "
                f"motion={motion_score:.2f}{ap}{grace}, n={st.frames})")
