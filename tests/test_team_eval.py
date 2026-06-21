"""TeamStats（GT 不要チーム分類プロキシ）の単体テスト。numpy のみで動く。

Run:  python tests/test_team_eval.py
  or: python -m pytest tests/test_team_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from alt_pix.team_eval import TeamStats


def test_consistent_teams_high_purity_no_flips():
    """各 track が常に同じチーム → purity=1, flip=0, balance 3:3。"""
    st = TeamStats()
    for f in range(100):
        st.update({1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1})
    s = st.summary()
    assert s["labeled_tracks"] == 6
    assert s["purity_mean"] == 1.0
    assert s["purity_weighted"] == 1.0
    assert s["flip_rate_per100"] == 0.0
    assert s["balance_mean0"] == 3.0
    assert s["balance_mean1"] == 3.0
    assert s["balance_ratio"] == 1.0


def test_flipping_track_lowers_purity_and_raises_flips():
    """1 つの track が毎フレーム反転 → purity 低下・flip 多発。"""
    st = TeamStats()
    for f in range(100):
        st.update({1: f % 2})  # 0,1,0,1,...
    s = st.summary()
    assert s["labeled_tracks"] == 1
    # 最頻ラベルは 50 フレーム → purity 0.5
    assert abs(s["purity_weighted"] - 0.5) < 1e-9
    # 99 回反転 / 100 フレーム
    assert abs(s["flip_rate_per100"] - 99.0) < 1e-9


def test_unassigned_label_ignored():
    """-1（warm-up 中）は集計対象外。"""
    st = TeamStats()
    for f in range(10):
        st.update({1: -1, 2: -1})
    s = st.summary()
    assert s["labeled_tracks"] == 0
    assert s["purity_mean"] == 0.0
    assert s["balance_mean0"] == 0.0


def test_balance_skew_detected():
    """全員 team0 に吸い込まれる → balance ratio が極端に低い。"""
    st = TeamStats()
    for f in range(50):
        st.update({1: 0, 2: 0, 3: 0, 4: 0, 5: 1})  # 4:1
    s = st.summary()
    assert s["balance_mean0"] == 4.0
    assert s["balance_mean1"] == 1.0
    assert abs(s["balance_ratio"] - 0.25) < 1e-9


def test_margin_stats_and_low_ratio():
    """margin の集計と低margin率。"""
    st = TeamStats()
    # 2 track: 高margin と低margin（曖昧）。
    for f in range(10):
        st.update({1: 0, 2: 1}, {1: 0.8, 2: 0.01})
    s = st.summary()
    assert abs(s["margin_mean"] - 0.405) < 1e-6
    assert abs(s["margin_low_ratio"] - 0.5) < 1e-9  # 半分が <0.05


def test_purity_histogram_buckets():
    st = TeamStats()
    # track 1: 一貫（purity 1.0）, track 2: 反転気味（purity 0.5）
    for f in range(100):
        st.update({1: 0, 2: f % 2})
    hist = dict(st.purity_histogram())
    assert hist["0.95-1.00"] == 1   # track 1
    assert hist["0.00-0.60"] == 1   # track 2
    assert sum(hist.values()) == 2


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
