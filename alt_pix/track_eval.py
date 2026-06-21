"""トラッキング品質のプロキシ評価（ground-truth 不要）。

実映像にアノテーションが無い段階で MOT を評価するため、正解ラベル不要で
ID 安定性を近似する指標を集計する。MOTA/IDF1 のような GT 必須指標は使わず、
以下の「壊れ方」が表面化するプロキシに絞る：

  - active_count   : 各フレームの確定 track 数（時系列）。乱高下＝検出/追従の不安定。
  - unique_ids     : 累計の新規 ID 数。実人数に対して過大なら fragmentation。
  - lifetime       : 各 ID の生存フレーム数の分布。短命が多い＝ID switch/分断。
  - short_ratio    : しきい値未満で消える track の割合（fragmentation 指標）。
  - new_id_rate    : 100 フレームあたりの新規 ID 発行数。低いほど安定。

これらは GT が無くても「ID がコロコロ変わる」「track が湧いては消える」状態を
定量化でき、tracker のパラメータ（activation_thresh / lost_buffer /
min_consecutive_frames）チューニングの良し悪しを比較できる。

純 Python / numpy のみ（onnxruntime 等の重依存なし）でユニットテスト可能。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class _TrackLife:
    first_frame: int
    last_frame: int
    n_frames: int = 0


@dataclass
class TrackingStats:
    """フレームごとの track_id 列を投入し、終了時に安定性指標を出す集計器。

    short_threshold: この値（フレーム）未満で消えた track を「短命」とみなす。
    """

    short_threshold: int = 5

    _lives: dict[int, _TrackLife] = field(default_factory=dict)
    _count_timeline: list[tuple[int, int]] = field(default_factory=list)
    _new_ids_per_frame: list[int] = field(default_factory=list)
    _n_frames: int = 0

    def update(self, frame_id: int, track_ids: list[int]) -> None:
        """1 フレーム分の確定 track ID 列を投入する。"""
        self._n_frames += 1
        n_new = 0
        for tid in track_ids:
            if tid not in self._lives:
                self._lives[tid] = _TrackLife(first_frame=frame_id, last_frame=frame_id)
                n_new += 1
            life = self._lives[tid]
            life.last_frame = frame_id
            life.n_frames += 1
        self._count_timeline.append((frame_id, len(track_ids)))
        self._new_ids_per_frame.append(n_new)

    # ── 集計 ─────────────────────────────────────────────────────────────────

    def lifetimes(self) -> np.ndarray:
        """各 ID の生存フレーム数（観測されたフレーム数）。"""
        return np.array([l.n_frames for l in self._lives.values()], dtype=int)

    def counts(self) -> np.ndarray:
        """各フレームの track 数の時系列。"""
        return np.array([c for _, c in self._count_timeline], dtype=int)

    def summary(self) -> dict:
        life = self.lifetimes()
        counts = self.counts()
        n_unique = len(self._lives)
        short = int((life < self.short_threshold).sum()) if len(life) else 0
        total_new = int(sum(self._new_ids_per_frame))
        return {
            "frames": self._n_frames,
            "unique_ids": n_unique,
            "active_mean": float(counts.mean()) if len(counts) else 0.0,
            "active_std": float(counts.std()) if len(counts) else 0.0,
            "active_min": int(counts.min()) if len(counts) else 0,
            "active_max": int(counts.max()) if len(counts) else 0,
            "lifetime_mean": float(life.mean()) if len(life) else 0.0,
            "lifetime_median": float(np.median(life)) if len(life) else 0.0,
            "lifetime_max": int(life.max()) if len(life) else 0,
            "short_lived": short,
            "short_ratio": (short / n_unique) if n_unique else 0.0,
            "new_id_rate_per100": (total_new / self._n_frames * 100) if self._n_frames else 0.0,
        }

    def lifetime_histogram(self, bins: tuple[int, ...] = (1, 5, 15, 30, 60, 120)) -> list[tuple[str, int]]:
        """生存フレーム数のヒストグラム（[lo, hi) のバケット）。"""
        life = self.lifetimes()
        edges = list(bins) + [10 ** 9]
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            label = f"{lo}-{hi-1}" if hi < 10 ** 9 else f"{lo}+"
            out.append((label, int(((life >= lo) & (life < hi)).sum())))
        return out

    def format_report(self) -> str:
        s = self.summary()
        lines = [
            "── トラッキング安定性（GT 不要プロキシ）──",
            f"  frames            : {s['frames']}",
            f"  unique track IDs  : {s['unique_ids']}",
            f"  active tracks/フレーム : mean={s['active_mean']:.1f} "
            f"std={s['active_std']:.1f} min={s['active_min']} max={s['active_max']}",
            f"  lifetime [frames] : mean={s['lifetime_mean']:.1f} "
            f"median={s['lifetime_median']:.0f} max={s['lifetime_max']}",
            f"  short-lived (<{self.short_threshold}f): {s['short_lived']} "
            f"({s['short_ratio']*100:.1f}% of IDs)  ← fragmentation 指標",
            f"  new-ID rate       : {s['new_id_rate_per100']:.1f} / 100 frames",
            "",
            "  lifetime histogram:",
        ]
        for label, cnt in self.lifetime_histogram():
            bar = "█" * min(cnt, 40)
            lines.append(f"    {label:>7s}f: {cnt:4d} {bar}")
        lines += [
            "",
            "  読み方:",
            "   - active が安定（std 小）で実人数に近い → 検出/追従が安定",
            "   - unique_ids が実人数に対し過大 / short_ratio 高 → ID switch・分断が多い",
            "   - new-ID rate 高 → track が湧いては消える（buffer 不足やしきい値が高い）",
        ]
        return "\n".join(lines)
