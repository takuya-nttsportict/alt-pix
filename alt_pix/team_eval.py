"""チーム分類品質のプロキシ評価（ground-truth 不要）。

選手のチーム正解ラベルが無い段階で TeamClassifier を評価するため、正解不要で
「分類が壊れている」状態を表面化させる指標を集計する。track_eval と同じ哲学。

前提となる物理: 同一 track（同一選手）は試合中チームを変えない。したがって
ラベルが揺れる＝誤分類。これを軸に以下を集計する。

  - team_purity   : track ごとに最頻ラベルへ一致するフレーム率（全 track 加重平均）。
                    1.0 に近いほど「各選手のチームが一貫」。
  - flip_rate     : track ごとのラベル変化回数 /100 フレーム。低いほど安定。
  - balance       : フレームあたり team0/team1 人数。バレーは 6:6 が基本なので
                    極端な偏りは「片チームへ吸い込まれている」誤分類のサイン。
  - margin        : 割当の確信度（最近 vs 次点セントロイド距離の正規化差）の分布。
                    二峰で高い側に山＝自信を持って分離。谷に集中＝曖昧。

margin は TeamClassifier.assignment_margin 由来の値を track 単位で投入する。
純 numpy のみでユニットテスト可能（onnxruntime / torch 不要）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class _TeamHist:
    counts: dict[int, int] = field(default_factory=dict)  # team -> frames
    last_label: int | None = None
    flips: int = 0
    n_frames: int = 0


@dataclass
class TeamStats:
    """フレームごとに {track_id: team(0/1, -1=未確定)} と任意の margin を投入する集計器。

    -1（warm-up 中／未割当）は無視し、確定ラベルのみ集計する。
    """

    _hist: dict[int, _TeamHist] = field(default_factory=dict)
    _balance: list[tuple[int, int]] = field(default_factory=list)  # (n_team0, n_team1)
    _margins: list[float] = field(default_factory=list)
    _n_frames: int = 0

    def update(
        self,
        team_map: dict[int, int],
        margin_map: dict[int, float] | None = None,
    ) -> None:
        """1 フレーム分のチーム割当（と任意の margin）を投入する。"""
        self._n_frames += 1
        n0 = n1 = 0
        for tid, team in team_map.items():
            if team < 0:
                continue
            if team == 0:
                n0 += 1
            elif team == 1:
                n1 += 1
            h = self._hist.setdefault(tid, _TeamHist())
            h.counts[team] = h.counts.get(team, 0) + 1
            h.n_frames += 1
            if h.last_label is not None and h.last_label != team:
                h.flips += 1
            h.last_label = team
        # 確定ラベルが存在するフレームのみ balance に積む（warm-up 中を除外）。
        if n0 or n1:
            self._balance.append((n0, n1))
        if margin_map:
            self._margins.extend(
                m for tid, m in margin_map.items() if team_map.get(tid, -1) >= 0
            )

    # ── 集計 ─────────────────────────────────────────────────────────────────

    def _purities(self) -> np.ndarray:
        """track ごとの最頻ラベル一致率。"""
        out = []
        for h in self._hist.values():
            if h.n_frames == 0:
                continue
            out.append(max(h.counts.values()) / h.n_frames)
        return np.array(out, dtype=float)

    def summary(self) -> dict:
        pur = self._purities()
        # フレーム加重の purity（長命 track を重く）。
        weighted_num = sum(max(h.counts.values()) for h in self._hist.values() if h.n_frames)
        weighted_den = sum(h.n_frames for h in self._hist.values())
        total_flips = sum(h.flips for h in self._hist.values())
        labeled_frames = sum(h.n_frames for h in self._hist.values())
        bal = np.array(self._balance, dtype=float)
        margins = np.array(self._margins, dtype=float)
        return {
            "frames": self._n_frames,
            "labeled_tracks": len(self._hist),
            "purity_mean": float(pur.mean()) if len(pur) else 0.0,
            "purity_weighted": (weighted_num / weighted_den) if weighted_den else 0.0,
            "purity_min": float(pur.min()) if len(pur) else 0.0,
            "flip_rate_per100": (total_flips / labeled_frames * 100) if labeled_frames else 0.0,
            "balance_mean0": float(bal[:, 0].mean()) if len(bal) else 0.0,
            "balance_mean1": float(bal[:, 1].mean()) if len(bal) else 0.0,
            "balance_ratio": (
                float(min(bal[:, 0].mean(), bal[:, 1].mean())
                      / max(bal[:, 0].mean(), bal[:, 1].mean(), 1e-9))
                if len(bal) else 0.0
            ),
            "margin_mean": float(margins.mean()) if len(margins) else 0.0,
            "margin_median": float(np.median(margins)) if len(margins) else 0.0,
            "margin_low_ratio": (
                float((margins < 0.05).mean()) if len(margins) else 0.0
            ),
        }

    def purity_histogram(
        self, bins: tuple[float, ...] = (0.0, 0.6, 0.8, 0.9, 0.95, 1.0)
    ) -> list[tuple[str, int]]:
        pur = self._purities()
        edges = list(bins)
        out = []
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            label = f"{lo:.2f}-{hi:.2f}"
            # 最上位バケットは右端（purity=1.0）を含める。
            if i == len(edges) - 2:
                mask = (pur >= lo) & (pur <= hi)
            else:
                mask = (pur >= lo) & (pur < hi)
            out.append((label, int(mask.sum())))
        return out

    def format_report(self) -> str:
        s = self.summary()
        lines = [
            "── チーム分類品質（GT 不要プロキシ）──",
            f"  frames              : {s['frames']}",
            f"  labeled tracks      : {s['labeled_tracks']}",
            f"  team purity         : mean={s['purity_mean']:.3f} "
            f"weighted={s['purity_weighted']:.3f} min={s['purity_min']:.3f}",
            f"  flip rate           : {s['flip_rate_per100']:.2f} / 100 frames",
            f"  team balance/frame  : team0={s['balance_mean0']:.1f} "
            f"team1={s['balance_mean1']:.1f}  ratio={s['balance_ratio']:.2f}",
            f"  assignment margin   : mean={s['margin_mean']:.3f} "
            f"median={s['margin_median']:.3f}  低margin(<0.05)率={s['margin_low_ratio']*100:.1f}%",
            "",
            "  per-track purity histogram:",
        ]
        for label, cnt in self.purity_histogram():
            bar = "█" * min(cnt, 40)
            lines.append(f"    {label}: {cnt:4d} {bar}")
        lines += [
            "",
            "  読み方:",
            "   - purity 高（>0.9）→ 各選手のチームが一貫（誤分類で揺れていない）",
            "   - flip rate 高 → 同一選手のラベルが頻繁に入れ替わる（特徴が不安定）",
            "   - balance ratio が極端に低い → 片チームへ吸い込まれている疑い",
            "   - 低margin 率が高い → セントロイド間が近く曖昧（warm-up/refit 不足や",
            "     ユニフォーム色が似ている、審判混入など）",
        ]
        return "\n".join(lines)
