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
    # Asymmetric on/off-court membership (fast-in, slow-out).
    on_court_now: bool = False      # inside the court THIS frame
    was_on_court: bool = False      # has ever been inside the court
    frames_since_exit: int = 999    # frames since last leaving the court (999 = never inside)
    assume_server: bool = False     # exited across an END line (u-axis) -> likely a server
    recently_exited: bool = False   # within the (direction-dependent) grace window after exit


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
    # Asymmetric grace after leaving the court (fast-in / slow-out): only players
    # ever enter the court, so we keep "field" long after they step out.
    grace_frames: int = 150          # general exit (sideline / ball chase) ~5s@30fps
    serve_grace_frames: int = 1800   # END-line exit (assume server) ~60s@30fps
    # Game-progress: with few people on court the game is paused (timeout / set
    # break / between rallies with everyone off). Used to gate framing.
    min_active_on_court: float = 4.0
    active_alpha: float = 0.05

    _states: dict[int, _PartState] = field(default_factory=dict)
    _on_court_count_ema: float = 0.0
    _frames_total: int = 0

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
            x1, y1, x2, y2 = t.bbox
            foot = ((x1 + x2) / 2.0, float(y2))
            on_court_now = self.court.is_on_court(t.bbox, self.on_court_margin)
            on = 1.0 if on_court_now else 0.0
            st.on_court_ema += self.alpha * (on - st.on_court_ema)
            st.on_court_now = on_court_now

            # ── asymmetric grace (fast-in / slow-out) ─────────────────────────
            # Only players enter the court, so when someone steps out we keep
            # the "field" benefit of the doubt for a while. If they crossed an
            # END line (court length / u-axis) it is very likely a serve, so the
            # grace is much longer (assume_server); a sideline exit (ball chase)
            # gets the shorter general grace.
            if on_court_now:
                st.frames_since_exit = 0
                st.was_on_court = True
                st.assume_server = False
                st.recently_exited = False
            elif st.was_on_court:
                if st.frames_since_exit == 0:  # the exit frame: classify direction
                    st.assume_server = self._exited_via_endline(foot)
                st.frames_since_exit += 1
                window = self.serve_grace_frames if st.assume_server else self.grace_frames
                st.recently_exited = st.frames_since_exit <= window

            # ── motion (bbox-height-normalised foot displacement) ─────────────
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

        # ── game-progress: EMA of how many people are currently on court ──────
        n_on = sum(1 for s in out.values() if s.on_court_now)
        self._frames_total += 1
        self._on_court_count_ema += self.active_alpha * (n_on - self._on_court_count_ema)
        return out

    # ── game-progress accessors ──────────────────────────────────────────────

    @property
    def on_court_count_ema(self) -> float:
        """Smoothed number of people currently inside the court."""
        return self._on_court_count_ema

    @property
    def game_active(self) -> bool:
        """False when too few people are on court (timeout / set break / dead time).

        Withheld (treated active) until enough frames have been seen to avoid a
        spurious "paused" verdict during warm-up.
        """
        if self._frames_total < self.min_frames:
            return True
        return self._on_court_count_ema >= self.min_active_on_court

    def _exited_via_endline(self, foot: tuple[float, float]) -> bool:
        """True if the foot is outside an END line (u-axis) more than a sideline.

        Court metres: u in [0,18] is the length (end lines at u=0 / u=18, net at
        u=9); v in [0,9] is the width (sidelines at v=0 / v=9). A serve happens
        behind an end line, so an exit dominated by the u-axis -> assume server.
        """
        try:
            u, v = self.court.image_to_court(foot[0], foot[1])
        except Exception:  # pragma: no cover - degenerate homography
            return False
        du = max(0.0 - u, u - 18.0, 0.0)   # how far past an end line
        dv = max(0.0 - v, v - 9.0, 0.0)    # how far past a sideline
        return du > 0.0 and du >= dv

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
        if st.recently_exited:
            tag = "server?" if st.assume_server else "exit"
            grace = f" {tag}+{st.frames_since_exit}f"
        else:
            grace = ""
        return (f"score={st.score:.2f} (oncourt={st.on_court_ema:.2f} "
                f"motion={motion_score:.2f}{ap}{grace}, n={st.frames})")
