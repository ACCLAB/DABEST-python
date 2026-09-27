"""Tests for the cluster-aware bootstrap and permutation test (`cluster_col`)."""

import numpy as np
import pandas as pd
import pytest

from dabest._api import load
from dabest._effsize_objects import TwoGroupsEffectSize, PermutationTest
from dabest._stats_tools import confint_2group_diff as ci2g


def make_clustered_data(n_participants=12, n_sets=4, sd_participant=1.5, seed=3):
    """
    Within-subject design in long format: every participant contributes
    `n_sets` paired sets of observations across three levels, so that the
    pairs (identified by `pair`) are nested within participants (`ID`).
    """
    rng = np.random.default_rng(seed)
    levels = ["L1", "L2", "L3"]
    rows = []
    for p in range(n_participants):
        u_p = rng.normal(0, sd_participant)  # participant-specific sensitivity
        for s in range(n_sets):
            u_s = rng.normal(0, 0.5)
            for k, level in enumerate(levels):
                rows.append(
                    dict(
                        ID="P{:02d}".format(p),
                        pair=p * n_sets + s,
                        Level=level,
                        Y=3 + 0.4 * k * (1 + u_p) + u_s + rng.normal(0, 0.5),
                    )
                )
    return pd.DataFrame(rows).sort_values(["pair", "Level"]).reset_index(drop=True)


DF = make_clustered_data()
PAIRED_KWARGS = dict(
    idx=("L1", "L2", "L3"),
    x="Level",
    y="Y",
    paired="sequential",
    id_col="pair",
    resamples=500,
    random_seed=11,
)


@pytest.fixture(scope="module")
def naive():
    return load(DF, **PAIRED_KWARGS)


@pytest.fixture(scope="module")
def clustered():
    return load(DF, cluster_col="ID", **PAIRED_KWARGS)


# ---------------------------------------------------------------------------
# Low-level resampling machinery
# ---------------------------------------------------------------------------
def test_cluster_codes_pool_labels_across_groups():
    (c0, c1), n = ci2g.cluster_codes(["a", "b", "a"], ["b", "c"])
    assert n == 3
    assert c0.tolist() == [0, 1, 0]
    assert c1.tolist() == [1, 2]

    with pytest.raises(ValueError):
        ci2g.cluster_codes(["a", None, "b"])


def test_cluster_tables_and_expand_cluster_draw():
    codes = np.array([2, 0, 2, 1, 0])
    offsets, members = ci2g.cluster_tables(codes, 3)
    assert offsets.tolist() == [0, 2, 3, 5]
    assert members[offsets[0] : offsets[1]].tolist() == [1, 4]
    assert members[offsets[2] : offsets[3]].tolist() == [0, 2]

    draw = np.array([2, 2, 1], dtype=np.int64)
    idx = ci2g.expand_cluster_draw(draw, offsets, members)
    assert idx.tolist() == [0, 2, 0, 2, 3]


def test_cluster_strata_follow_the_design():
    # Fully within-cluster: every cluster in both groups -> one stratum.
    strata, offsets = ci2g.cluster_strata(([0, 1, 2], [0, 1, 2]), 3)
    assert offsets.tolist() == [0, 3]

    # Nested: clusters 0-1 only in control, 2-3 only in test -> one stratum per group.
    strata, offsets = ci2g.cluster_strata(([0, 1], [2, 3]), 4)
    assert offsets.tolist() == [0, 2, 4]
    assert sorted(strata[:2].tolist()) == [0, 1]
    assert sorted(strata[2:].tolist()) == [2, 3]

    # Mixed: one stratum per membership pattern.
    strata, offsets = ci2g.cluster_strata(([0, 1, 2], [1, 2, 3]), 4)
    assert offsets.tolist() == [0, 1, 2, 4]


def test_cluster_bootstrap_reduces_to_ordinary_bootstrap_for_singleton_clusters():
    rng = np.random.default_rng(1)
    x0 = rng.normal(0, 1, 15)
    x1 = rng.normal(0.5, 1, 15)
    ids = np.arange(15)

    # Paired: every pair is its own cluster.
    plain = ci2g.compute_bootstrapped_diff(x0, x1, "baseline", "mean_diff", 300, 7)
    clus = ci2g.compute_cluster_bootstrapped_diff(x0, x1, ids, ids, "baseline", "mean_diff", 300, 7)
    assert np.array_equal(plain, clus)

    # Unpaired: every observation is its own cluster (nested design).
    x1u = rng.normal(0.5, 1, 20)
    plain = ci2g.compute_bootstrapped_diff(x0, x1u, None, "mean_diff", 300, 7)
    clus = ci2g.compute_cluster_bootstrapped_diff(
        x0, x1u, np.arange(15), 100 + np.arange(20), None, "mean_diff", 300, 7
    )
    assert np.array_equal(plain, clus)

    # The delete-one jackknife also coincides.
    plain = ci2g.compute_meandiff_jackknife(x0, x1, "baseline", "mean_diff")
    clus = ci2g.compute_cluster_jackknife(x0, x1, ids, ids, "baseline", "mean_diff")
    assert plain == pytest.approx(clus)


def test_cluster_bootstrap_keeps_pairs_together():
    # With a constant within-pair difference in every cluster, every paired
    # cluster-bootstrap resample must reproduce that difference exactly.
    x0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    x1 = x0 + np.array([0.5, 0.5, 1.0, 1.0, 2.0, 2.0])
    clusters = np.array([0, 0, 1, 1, 2, 2])
    boots = ci2g.compute_cluster_bootstrapped_diff(x0, x1, clusters, clusters, "baseline", "mean_diff", 200, 3)
    # Three equally sized clusters are drawn with replacement, so every
    # resample's mean difference is the mean of three cluster differences.
    from itertools import combinations_with_replacement

    allowed = {round(np.mean(c), 10) for c in combinations_with_replacement([0.5, 1.0, 2.0], 3)}
    assert set(np.round(boots, 10)).issubset(allowed)


def test_cluster_bootstrap_rejects_misaligned_paired_clusters():
    x = np.arange(6, dtype=float)
    with pytest.raises(ValueError, match="same cluster"):
        ci2g.compute_cluster_bootstrapped_diff(x, x, [0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 0], "baseline", "mean_diff", 10, 1)


# ---------------------------------------------------------------------------
# dabest.load with cluster_col
# ---------------------------------------------------------------------------
def test_point_estimates_are_unchanged_by_clustering(naive, clustered):
    for effect_size in ["mean_diff", "cohens_d", "hedges_g"]:
        n = getattr(naive, effect_size).results["difference"].to_numpy()
        c = getattr(clustered, effect_size).results["difference"].to_numpy()
        assert n == pytest.approx(c)


def test_cluster_intervals_are_wider_under_participant_effects(naive, clustered):
    rn = naive.mean_diff.results
    rc = clustered.mean_diff.results
    sd_ratio = [np.std(c) / np.std(n) for c, n in zip(rc["bootstraps"], rn["bootstraps"])]
    assert all(r > 1.2 for r in sd_ratio)
    assert ((rc["bca_high"] - rc["bca_low"]) > (rn["bca_high"] - rn["bca_low"])).all()


def test_results_report_clusters(naive, clustered):
    rc = clustered.mean_diff.results
    assert rc["n_clusters"].tolist() == [12, 12]
    assert "n_clusters" not in naive.mean_diff.results.columns

    assert clustered.cluster_col == "ID"
    assert naive.cluster_col is None
    assert "clusters" in repr(clustered)
    assert "cluster" in repr(clustered.mean_diff)
    assert "cluster" not in repr(naive.mean_diff)


def test_baseline_pairing_with_clusters():
    result = load(DF, cluster_col="ID", **dict(PAIRED_KWARGS, paired="baseline")).mean_diff.results
    assert result["n_clusters"].tolist() == [12, 12]
    assert result["control"].tolist() == ["L1", "L1"]


def test_unpaired_with_clusters_spanning_both_groups():
    kwargs = dict(idx=("L1", "L3"), x="Level", y="Y", resamples=300, random_seed=5)
    naive = load(DF, **kwargs).mean_diff.results
    clustered = load(DF, cluster_col="ID", **kwargs).mean_diff.results
    assert clustered["n_clusters"].iloc[0] == 12
    assert clustered["difference"].iloc[0] == pytest.approx(naive["difference"].iloc[0])
    assert 0 <= clustered["pvalue_permutation"].iloc[0] <= 1
    assert len(clustered["permutations"].iloc[0]) == 5000


def test_unpaired_with_clusters_nested_within_groups():
    # Participants P00-P05 measured at L1 only; P06-P11 at L3 only.
    df = DF[DF["Level"].isin(["L1", "L3"])]
    first_half = df["ID"] < "P06"
    df = df[(first_half & (df["Level"] == "L1")) | (~first_half & (df["Level"] == "L3"))]
    result = load(
        df, idx=("L1", "L3"), x="Level", y="Y", cluster_col="ID", resamples=300, ps_adjust=True
    ).mean_diff.results
    assert result["n_clusters"].iloc[0] == 12
    assert result["control_N"].iloc[0] == 24
    assert 0 <= result["pvalue_permutation"].iloc[0] <= 1


def test_wide_format_with_clusters():
    wide = DF.pivot(index="pair", columns="Level", values="Y").reset_index()
    wide["ID"] = wide["pair"] // 4
    result = load(
        wide, idx=("L1", "L2", "L3"), paired="sequential", id_col="pair", cluster_col="ID", resamples=300
    ).mean_diff.results
    long_result = load(DF, cluster_col="ID", **dict(PAIRED_KWARGS, resamples=300)).mean_diff.results
    assert result["n_clusters"].tolist() == [12, 12]
    assert result["difference"].to_numpy() == pytest.approx(long_result["difference"].to_numpy())


def test_delta2_and_mini_meta_with_clusters():
    df = DF[DF["Level"].isin(["L1", "L3"])].copy()
    df["Env"] = np.where(df["pair"] % 4 < 2, "A", "B")

    delta2 = load(
        df, x=["Level", "Env"], y="Y", delta2=True, experiment="Env",
        paired="sequential", id_col="pair", cluster_col="ID", resamples=300,
    )
    dd = delta2.mean_diff.delta_delta
    assert dd.bca_low < dd.difference < dd.bca_high
    assert len(dd.bootstraps_delta_delta) == 300
    dg = delta2.hedges_g.delta_delta
    assert dg.bca_low < dg.difference < dg.bca_high

    mini_meta = load(
        df, idx=(("L1", "L3"),), x="Level", y="Y", mini_meta=True,
        paired="sequential", id_col="pair", cluster_col="ID", resamples=300,
    )
    mm = mini_meta.mean_diff.mini_meta
    assert mm.bca_low < mm.difference < mm.bca_high


def test_other_effect_sizes_and_plot_with_clusters(clustered):
    import matplotlib

    matplotlib.use("Agg")
    assert len(clustered.median_diff.results) == 2
    fig = clustered.mean_diff.plot()
    assert fig is not None


def test_plot_tick_labels_report_cluster_count(naive, clustered):
    import matplotlib

    matplotlib.use("Agg")

    naive_labels = [t.get_text() for t in naive.mean_diff.plot().axes[0].get_xticklabels()]
    for label in naive_labels:
        assert "n=" not in label
        assert "(N=" in label

    clustered_labels = [t.get_text() for t in clustered.mean_diff.plot().axes[0].get_xticklabels()]
    for label in clustered_labels:
        assert "(N=48,\n n=12)" in label  # 12 participants x 4 sets each = 48 observations per level

    # A design where each participant contributes several sets: the cluster
    # count in the label must be lower than the observation count.
    df = make_clustered_data(n_participants=6, n_sets=5)
    clustered_multi = load(
        df, idx=("L1", "L2", "L3"), x="Level", y="Y", paired="sequential",
        id_col="pair", cluster_col="ID", resamples=200, random_seed=1,
    )
    labels = [t.get_text() for t in clustered_multi.mean_diff.plot().axes[0].get_xticklabels()]
    for label in labels:
        assert "(N=30,\n n=6)" in label

    # Horizontal orientation keeps the group label and N on one line, but still
    # breaks N and n onto separate lines.
    horiz_labels = [t.get_text() for t in clustered.mean_diff.plot(horizontal=True).axes[0].get_yticklabels()]
    for label in horiz_labels:
        assert " (N=48,\n n=12)" in label


def test_cumming_layout_makes_room_for_cluster_label(naive, clustered):
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    def gap(fig):
        raw, contrast = fig.axes[0].get_position(), fig.axes[1].get_position()
        return raw.y0 - contrast.y1

    # Figures created by dabest.
    naive_gap = gap(naive.mean_diff.plot(float_contrast=False))
    clustered_gap = gap(clustered.mean_diff.plot(float_contrast=False))
    assert clustered_gap > naive_gap

    # Figures drawn into a user-supplied axes (the contrast axes is an inset,
    # placed when the figure is drawn).
    def inset_gap(dabest_obj):
        f, ax = plt.subplots()
        dabest_obj.mean_diff.plot(ax=ax, float_contrast=False)
        f.canvas.draw()
        renderer = f.canvas.get_renderer()
        return ax.get_window_extent(renderer).y0 - ax.contrast_axes.get_window_extent(renderer).y1

    assert inset_gap(clustered) > inset_gap(naive)

    # Without sample-size labels there is nothing extra to make room for.
    assert gap(clustered.mean_diff.plot(float_contrast=False, show_sample_size=False)) == pytest.approx(naive_gap)
    plt.close("all")


def _label_clearance(raw_ax, contrast_ax):
    """Pixels between the lowest raw-data tick label and the top of the contrast axes."""
    fig = raw_ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = [t for t in raw_ax.get_xticklabels() if t.get_text()]
    label_bottom = min(t.get_window_extent(renderer).y0 for t in labels)
    return label_bottom - contrast_ax.get_window_extent(renderer).y1


def test_cluster_labels_clear_the_contrast_axes():
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")
    rng = np.random.default_rng(0)
    df = DF.copy()
    df["Yes"] = (rng.random(len(df)) < 0.4).astype(int)
    proportional_kwargs = dict(idx=("L1", "L2", "L3"), x="Level", y="Yes", proportional=True,
                               paired="sequential", id_col="pair", cluster_col="ID", resamples=200)

    cases = [
        (load(DF, cluster_col="ID", **PAIRED_KWARGS), dict(fig_size=(5, 4))),
        (load(DF, cluster_col="ID", **PAIRED_KWARGS), dict(fontsize_rawxlabel=16)),
        # Two-column Sankey labels span four lines before the cluster count is added.
        (load(df, **proportional_kwargs), dict(sankey_kwargs={"flow": False})),
        (load(df, **proportional_kwargs), dict(sankey_kwargs={"flow": False}, fig_size=(5, 4))),
    ]
    for dabest_obj, plot_kwargs in cases:
        fig = dabest_obj.mean_diff.plot(**plot_kwargs)
        assert _label_clearance(fig.axes[0], fig.axes[1]) > 0, plot_kwargs

        user_kwargs = {k: v for k, v in plot_kwargs.items() if k != "fig_size"}
        f, ax = plt.subplots(figsize=(5.5, 5))
        dabest_obj.mean_diff.plot(ax=ax, **user_kwargs)
        assert _label_clearance(ax, ax.contrast_axes) > 0, user_kwargs
        plt.close("all")


def test_gap_for_cluster_labels_survives_autolayout(naive, clustered):
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    def gap_points(dabest_obj):
        """Points between the raw data axes and the contrast axes, and below the tick labels."""
        f, ax = plt.subplots(figsize=(6.5, 3.5))
        dabest_obj.mean_diff.plot(ax=ax, float_contrast=False)
        clearance = _label_clearance(ax, ax.contrast_axes)
        renderer = f.canvas.get_renderer()
        gap = ax.get_window_extent(renderer).y0 - ax.contrast_axes.get_window_extent(renderer).y1
        plt.close("all")
        return gap * 72 / f.dpi, clearance * 72 / f.dpi

    # `figure.autolayout` resizes the user's axes after dabest has laid it out;
    # the gap for the cluster labels must not be stretched along with it.
    plain_gap, plain_clearance = gap_points(clustered)
    with matplotlib.rc_context({"figure.autolayout": True}):
        auto_gap, auto_clearance = gap_points(clustered)
        naive_gap, _ = gap_points(naive)
    assert auto_gap == pytest.approx(plain_gap, abs=1)
    assert 0 < auto_clearance < 20 and 0 < plain_clearance < 20
    assert auto_gap < 1.5 * naive_gap


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_cluster_col_validation():
    with pytest.raises(IndexError, match="not a column"):
        load(DF, cluster_col="missing", **PAIRED_KWARGS)

    with pytest.raises(ValueError, match="same column as `y`"):
        load(DF, cluster_col="Y", **PAIRED_KWARGS)

    bad = DF.copy()
    bad.loc[0, "ID"] = "P99"  # one row of pair 0 now belongs to another participant
    with pytest.raises(ValueError, match="single cluster"):
        load(bad, cluster_col="ID", **PAIRED_KWARGS)

    bad = DF.copy()
    bad.loc[0, "ID"] = None
    with pytest.raises(ValueError, match="missing values"):
        load(bad, cluster_col="ID", **PAIRED_KWARGS)

    wide = DF.pivot(index="pair", columns="Level", values="Y").reset_index()
    with pytest.raises(ValueError, match="one of the groups"):
        load(wide, idx=("L1", "L2"), paired="sequential", id_col="pair", cluster_col="L1")


def test_two_groups_effect_size_direct_api():
    rng = np.random.default_rng(2)
    control = rng.normal(0, 1, 8)
    test = rng.normal(0.5, 1, 8)
    clusters = np.repeat([0, 1, 2, 3], 2)

    result = TwoGroupsEffectSize(
        control, test, "mean_diff", is_paired="baseline", resamples=200,
        control_clusters=clusters, test_clusters=clusters,
    )
    assert result.is_clustered
    assert result.n_clusters == 4
    assert result.to_dict()["n_clusters"] == 4

    plain = TwoGroupsEffectSize(control, test, "mean_diff", is_paired="baseline", resamples=200)
    assert not plain.is_clustered
    assert plain.n_clusters is None

    with pytest.raises(ValueError, match="same lengths"):
        TwoGroupsEffectSize(control, test, "mean_diff", resamples=200, control_clusters=clusters[:4], test_clusters=clusters)

    with pytest.raises(ValueError, match="Both"):
        TwoGroupsEffectSize(control, test, "mean_diff", resamples=200, control_clusters=clusters)

    with pytest.raises(ValueError, match="same cluster"):
        TwoGroupsEffectSize(
            control, test, "mean_diff", is_paired="baseline", resamples=200,
            control_clusters=clusters, test_clusters=clusters[::-1],
        )


def test_cluster_permutation_test():
    rng = np.random.default_rng(4)
    control = rng.normal(0, 1, 12)
    test = control + 1.0
    clusters = np.repeat([0, 1, 2, 3], 3)

    # Paired: with a constant difference of 1 in every pair, each cluster-level
    # sign flip changes the mean difference by a multiple of 2 * 3 / 12.
    paired = PermutationTest(
        control, test, "mean_diff", is_paired="baseline", permutation_count=200,
        control_clusters=clusters, test_clusters=clusters,
    )
    assert 0 <= paired.pvalue <= 1
    assert set(np.round(paired.permutations, 10)).issubset({-1.0, -0.5, 0.0, 0.5, 1.0})

    # Unpaired with shared clusters, adjusted p-value.
    shared = PermutationTest(
        control, test, "mean_diff", permutation_count=200, ps_adjust=True,
        control_clusters=clusters, test_clusters=clusters,
    )
    assert 0 <= shared.pvalue <= 1
    assert len(shared.permutations_var) == 200

    # Shared clusters are permuted by swapping each cluster's control and test
    # sets as a whole. When the two sets of every cluster hold the same values,
    # no swap can change the effect size, whereas reshuffling observations
    # within clusters would.
    values = np.array([1.0, 5.0, 2.0, 8.0, 3.0, 9.0])
    sets = np.repeat([0, 1, 2], 2)
    identical = PermutationTest(
        values, values, "mean_diff", permutation_count=100,
        control_clusters=sets, test_clusters=sets,
    )
    assert np.allclose(identical.permutations, 0.0)

    # When every cluster's sets differ by a constant, the permutation effect
    # sizes can only take the values produced by whole-set swaps.
    shifted = PermutationTest(
        values, values + 2.0, "mean_diff", permutation_count=100,
        control_clusters=sets, test_clusters=sets,
    )
    allowed = {-2.0, -2 / 3, 2 / 3, 2.0}
    assert all(any(np.isclose(v, a) for a in allowed) for v in shifted.permutations)

    # Unpaired with nested clusters, adjusted p-value.
    nested = PermutationTest(
        control, test, "mean_diff", permutation_count=200, ps_adjust=True,
        control_clusters=clusters, test_clusters=clusters + 10,
    )
    assert 0 <= nested.pvalue <= 1


# ---------------------------------------------------------------------------
# Small-sample expansion of cluster-bootstrap intervals (`cluster_ci_expansion`)
# ---------------------------------------------------------------------------
def _hesterberg_level(n, ci=95, df=None):
    """
    Hesterberg's (2015) expanded percentile level for a single sample of n units;
    `df` replaces the n - 1 degrees of freedom of the t quantile.
    """
    from scipy import stats

    alpha = (100 - ci) / 100
    z = np.sqrt(n / (n - 1)) * stats.t.ppf(1 - alpha / 2, n - 1 if df is None else df)
    return 100 * (1 - 2 * stats.norm.sf(z))


def test_expanded_level_matches_hesterberg_for_one_stratum():
    for n in (3, 5, 8, 15, 30, 200):
        level, df = ci2g.expanded_ci_level(95, [(0.37, n)])
        assert level == pytest.approx(_hesterberg_level(n))
        assert df == pytest.approx(n - 1)
    # Any confidence level, and the result does not depend on the variance scale.
    assert ci2g.expanded_ci_level(90, [(5.0, 12)])[0] == pytest.approx(_hesterberg_level(12, ci=90))
    assert ci2g.expanded_ci_level(95, [(1e-9, 12)])[0] == pytest.approx(_hesterberg_level(12))


def test_expanded_level_behaviour():
    levels = [ci2g.expanded_ci_level(95, [(1.0, n)])[0] for n in (4, 8, 16, 32, 64, 1000)]
    assert all(a > b for a, b in zip(levels, levels[1:]))  # fewer clusters, more expansion
    assert levels[-1] > 95 and levels[-1] == pytest.approx(95, abs=0.05)

    # Two equal, independently resampled strata: the Welch-Satterthwaite degrees
    # of freedom pool both strata, so the expansion is smaller than for either alone.
    level, df = ci2g.expanded_ci_level(95, [(1.0, 8), (1.0, 8)])
    assert df == pytest.approx(14)
    assert level == pytest.approx(_hesterberg_level(8, df=14))
    assert 95 < level < _hesterberg_level(8)

    # Unequal strata: the degrees of freedom are weighted by the variance shares.
    _, df = ci2g.expanded_ci_level(95, [(1.0, 5), (3.0, 20)])
    assert df == pytest.approx(4 ** 2 / (1 / 4 + 3 ** 2 / 19))
    # A stratum carrying a negligible share of the variance barely counts ...
    level_dominated, df = ci2g.expanded_ci_level(95, [(1.0, 40), (1e-9, 3)])
    assert df == pytest.approx(39)
    assert level_dominated == pytest.approx(_hesterberg_level(40), rel=1e-4)
    # ... and a small stratum carrying most of it sets both corrections.
    level_dominated, df = ci2g.expanded_ci_level(95, [(1e-9, 40), (1.0, 6)])
    assert df == pytest.approx(5)
    assert level_dominated == pytest.approx(_hesterberg_level(6), rel=1e-4)
    # Many equal strata.
    level, df = ci2g.expanded_ci_level(95, [(1.0, 10)] * 12)
    assert df == pytest.approx(12 * 9) and level == pytest.approx(_hesterberg_level(10, df=108))

    # Whatever the variances, the degrees of freedom lie between those of the
    # smallest stratum and the pooled total.
    rng = np.random.default_rng(3)
    for _ in range(50):
        sizes = rng.integers(2, 30, size=rng.integers(2, 5))
        components = list(zip(rng.exponential(1, sizes.size), sizes))
        _, df = ci2g.expanded_ci_level(95, components)
        assert sizes.min() - 1 - 1e-9 <= df <= np.sum(sizes - 1) + 1e-9

    # Strata of a single cluster carry no information and are ignored.
    assert ci2g.expanded_ci_level(95, [(1.0, 1), (1.0, 10)])[0] == pytest.approx(_hesterberg_level(10))
    assert ci2g.expanded_ci_level(95, [(1.0, 1)]) == (95, None)
    assert ci2g.expanded_ci_level(95, []) == (95, None)

    # No usable variance information: strata are weighted equally.
    assert ci2g.expanded_ci_level(95, [(0.0, 8), (0.0, 6)])[1] == pytest.approx(4 / (1 / 7 + 1 / 5))

    # The smallest group of clusters that carries a real share of the variance,
    # used to warn about designs with too few clusters.
    assert ci2g.smallest_cluster_group([(1.0, 5), (3.0, 20)]) == 5
    assert ci2g.smallest_cluster_group([(1.0, 40), (1e-9, 3)]) == 40
    assert ci2g.smallest_cluster_group([(1.0, 10)] * 12) == 10
    assert ci2g.smallest_cluster_group([(1.0, 1)]) is None

    # Extreme expansion is floored so that quantile functions stay finite.
    level, _ = ci2g.expanded_ci_level(95, [(1.0, 2)])
    assert 100 - level == pytest.approx(100 * ci2g._MIN_EXPANDED_ALPHA)


def test_cluster_variance_components_follow_the_strata():
    x = np.arange(12, dtype=float)
    y = x + np.linspace(0, 1, 12)

    # Paired: one stratum holding every cluster.
    clusters = np.repeat(np.arange(4), 3)
    values, deleted, codes, n = ci2g.cluster_jackknife_by_cluster(x, y, clusters, clusters, "baseline", "mean_diff")
    components = ci2g.cluster_variance_components(values, deleted, codes, n)
    assert [n_s for _, n_s in components] == [4]
    assert components[0][0] > 0

    # Nested: one stratum per group.
    values, deleted, codes, n = ci2g.cluster_jackknife_by_cluster(
        x[:6], y[6:], np.repeat([0, 1, 2], 2), np.repeat([5, 6, 7], 2), None, "mean_diff")
    assert sorted(n_s for _, n_s in ci2g.cluster_variance_components(values, deleted, codes, n)) == [3, 3]

    # Mixed: clusters in both groups, and single-group clusters; a stratum of one
    # cluster is left out.
    c0 = np.array([0, 0, 1, 1, 2, 2, 3])
    c1 = np.array([0, 0, 1, 1, 2, 2, 4, 4])
    values, deleted, codes, n = ci2g.cluster_jackknife_by_cluster(x[:7], y[:8], c0, c1, None, "mean_diff")
    assert sorted(n_s for _, n_s in ci2g.cluster_variance_components(values, deleted, codes, n)) == [3]

    # The delta-delta counterpart.
    comps = ci2g.delta2_cluster_variance_components(x, y, x + 1, y + 2, clusters, clusters, clusters, clusters, "baseline")
    assert [n_s for _, n_s in comps] == [4]


def test_interval_index_helpers():
    assert ci2g.percentile_interval_idx(95, 1000) == (25, 975)
    assert ci2g.percentile_interval_idx(99.9999999999, 10) == (0, 9)

    # BCa at an expanded level: valid indexes, within the array.
    low, high = ci2g.expanded_interval_limits(0.05, 0.02, 1000, 98.0)
    assert 0 <= low < high <= 999
    # Undefined when a large acceleration meets an extreme level.
    low, high = ci2g.expanded_interval_limits(0.0, 0.5, 1000, 99.99)
    assert np.isnan(low) and np.isnan(high)
    low, high = ci2g.expanded_interval_limits(np.inf, 0.0, 1000, 98.0)
    assert np.isnan(low) and np.isnan(high)


def test_expansion_is_on_by_default_and_can_be_turned_off(clustered):
    expanded = clustered.mean_diff.results
    unexpanded = load(DF, cluster_col="ID", cluster_ci_expansion=False, **PAIRED_KWARGS).mean_diff.results

    # 12 participants, all present in every group: a single stratum.
    assert expanded["ci_expanded"].to_numpy() == pytest.approx(_hesterberg_level(12))
    assert (expanded["ci"] == 95).all()
    assert "ci_expanded" not in unexpanded.columns

    # Same point estimates and bootstrap distributions; only where they are read changes.
    assert expanded["difference"].to_numpy() == pytest.approx(unexpanded["difference"].to_numpy())
    for boot_on, boot_off in zip(expanded["bootstraps"], unexpanded["bootstraps"]):
        assert np.array_equal(boot_on, boot_off)

    for kind in ("bca", "pct", "bec_bca", "bec_pct"):
        width_on = expanded[f"{kind}_high"] - expanded[f"{kind}_low"]
        width_off = unexpanded[f"{kind}_high"] - unexpanded[f"{kind}_low"]
        assert (width_on > width_off).all(), kind

    # The percentile limits are the bootstrap quantiles at the reported levels.
    resamples = PAIRED_KWARGS["resamples"]
    for row_on, row_off in zip(expanded.itertuples(), unexpanded.itertuples()):
        on, off = np.sort(row_on.bootstraps), np.sort(row_off.bootstraps)
        a = (100 - row_on.ci_expanded) / 200
        assert row_on.pct_low == on[int(a * resamples)]
        assert row_on.pct_high == on[int((1 - a) * resamples)]
        assert row_off.pct_low == off[int(0.025 * resamples)]
        assert row_off.pct_high == off[int(0.975 * resamples)]

    assert "ci_expanded" in clustered.mean_diff.statistical_tests.columns
    assert "read at the" in repr(clustered.mean_diff)
    assert "cluster_ci_expansion=False" in repr(clustered)
    unexpanded_repr = repr(load(DF, cluster_col="ID", cluster_ci_expansion=False, **PAIRED_KWARGS).mean_diff)
    assert "read at the" not in unexpanded_repr


def test_unclustered_results_are_unaffected(naive):
    results = naive.mean_diff.results
    assert "ci_expanded" not in results.columns
    resamples = PAIRED_KWARGS["resamples"]
    for row in results.itertuples():
        boot = np.sort(row.bootstraps)
        assert row.pct_low == boot[int(0.025 * resamples)]
        assert row.pct_high == boot[int(0.975 * resamples)]
    # The switch only concerns clustered data.
    with_switch = load(DF, cluster_ci_expansion=False, **PAIRED_KWARGS).mean_diff.results
    assert with_switch["bca_low"].to_numpy() == pytest.approx(results["bca_low"].to_numpy())
    assert "Confidence intervals will be expanded" not in repr(naive)


def test_expansion_for_unpaired_nested_and_mixed_designs():
    kwargs = dict(x="Level", y="Y", resamples=500, random_seed=4)
    # Nested: participants P00-P05 at L1 only, P06-P11 at L3 only (two strata of 6).
    df = DF[DF["Level"].isin(["L1", "L3"])]
    first_half = df["ID"] < "P06"
    nested = df[(first_half & (df["Level"] == "L1")) | (~first_half & (df["Level"] == "L3"))]
    result = load(nested, idx=("L1", "L3"), cluster_col="ID", **kwargs).mean_diff.results.iloc[0]
    # Two strata of 6: the narrowness correction of 6 clusters, with between 5 and
    # 10 degrees of freedom depending on how the variance divides between them.
    assert _hesterberg_level(6, df=10) < result["ci_expanded"] < _hesterberg_level(6)

    # Mixed: a participant measured only in the test group adds a stratum of one,
    # which cannot be resampled and so does not change the expansion.
    extra = pd.DataFrame({"ID": ["P99"] * 4, "pair": range(900, 904), "Level": "L3", "Y": [5.0, 6.0, 7.0, 5.5]})
    shared = DF[DF["Level"].isin(["L1", "L3"])]
    base = load(shared, idx=("L1", "L3"), cluster_col="ID", **kwargs).mean_diff.results.iloc[0]
    mixed = load(pd.concat([shared, extra]), idx=("L1", "L3"), cluster_col="ID", **kwargs).mean_diff.results.iloc[0]
    assert mixed["ci_expanded"] == pytest.approx(base["ci_expanded"])


def test_expansion_for_delta2_and_mini_meta():
    df = DF[DF["Level"].isin(["L1", "L3"])].copy()
    df["Env"] = np.where(df["pair"] % 4 < 2, "A", "B")
    delta2_kwargs = dict(x=["Level", "Env"], y="Y", delta2=True, experiment="Env",
                         paired="sequential", id_col="pair", cluster_col="ID", resamples=500)
    mini_meta_kwargs = dict(idx=(("L1", "L3"),), x="Level", y="Y", mini_meta=True,
                            paired="sequential", id_col="pair", cluster_col="ID", resamples=500)

    for effect_size in ("mean_diff", "hedges_g"):
        on = getattr(load(df, **delta2_kwargs), effect_size).delta_delta
        off = getattr(load(df, cluster_ci_expansion=False, **delta2_kwargs), effect_size).delta_delta
        # Every participant contributes to all four groups: a single stratum of 12.
        assert on.ci_expanded == pytest.approx(_hesterberg_level(12))
        assert off.ci_expanded is None
        assert (on.bca_high - on.bca_low) > (off.bca_high - off.bca_low)
        assert (on.pct_high - on.pct_low) > (off.pct_high - off.pct_low)
        assert "ci_expanded" in on.results.columns and "ci_expanded" not in off.results.columns
        assert "read at the" in repr(on)

    on = load(df, **mini_meta_kwargs).mean_diff.mini_meta
    off = load(df, cluster_ci_expansion=False, **mini_meta_kwargs).mean_diff.mini_meta
    assert on.ci_expanded > 95 and off.ci_expanded is None
    assert (on.bca_high - on.bca_low) > (off.bca_high - off.bca_low)
    assert (on.pct_high - on.pct_low) > (off.pct_high - off.pct_low)
    assert "ci_expanded" in on.results.columns and "ci_expanded" not in off.results.columns
    assert "read at the" in repr(on) and "read at the" not in repr(off)


def test_expansion_edge_cases():
    import warnings

    rng = np.random.default_rng(7)
    # Two or three clusters: extreme but valid limits, and no index errors.
    for n_clusters in (2, 3):
        clusters = np.repeat(np.arange(n_clusters), 4)
        control = rng.normal(0, 1, clusters.size)
        test = control + 0.5 + rng.normal(0, 1, clusters.size)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = TwoGroupsEffectSize(control, test, "mean_diff", is_paired="baseline", resamples=300,
                                         permutation_count=10, control_clusters=clusters, test_clusters=clusters)
        assert result.ci_expanded > 99
        assert result.bca_low <= result.difference <= result.bca_high
        assert result.pct_low <= result.pct_high

    # Too few clusters to resample: the user is warned the interval is unreliable,
    # whether there are too few clusters overall ...
    clusters = np.repeat(np.arange(5), 4)
    control = rng.normal(0, 1, clusters.size)
    with pytest.warns(UserWarning, match="Only 5 clusters"):
        TwoGroupsEffectSize(control, control + 0.5 + rng.normal(0, 1, clusters.size), "mean_diff",
                            is_paired="baseline", resamples=300, permutation_count=10,
                            control_clusters=clusters, test_clusters=clusters)
    # ... or too few in one of the independently resampled groups of clusters.
    control, test = rng.normal(0, 1, 12), rng.normal(1, 1, 24)
    with pytest.warns(UserWarning, match=r"Only 9 clusters .*\(3 in the smallest"):
        TwoGroupsEffectSize(control, test, "mean_diff", resamples=300, permutation_count=10,
                            control_clusters=np.repeat(np.arange(3), 4),
                            test_clusters=np.repeat(np.arange(10, 16), 4))
    # Enough clusters: no such warning.
    clusters = np.repeat(np.arange(6), 4)
    control = rng.normal(0, 1, clusters.size)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        TwoGroupsEffectSize(control, control + 0.5 + rng.normal(0, 1, clusters.size), "mean_diff",
                            is_paired="baseline", resamples=1000, permutation_count=10,
                            control_clusters=clusters, test_clusters=clusters)
    assert not any("available to resample" in str(w.message) for w in caught)

    # One cluster per group: nothing can be expanded, and the user is told so.
    control, test = rng.normal(0, 1, 5), rng.normal(1, 1, 5)
    with pytest.warns(UserWarning, match="too few clusters"):
        result = TwoGroupsEffectSize(control, test, "mean_diff", resamples=200, permutation_count=10,
                                     control_clusters=np.zeros(5), test_clusters=np.ones(5))
    assert result.ci_expanded is None

    # Opt-out on the direct API, and argument validation.
    clusters = np.repeat(np.arange(6), 2)
    control = rng.normal(0, 1, 12)
    result = TwoGroupsEffectSize(control, control + 1, "mean_diff", is_paired="baseline", resamples=200,
                                 permutation_count=10, control_clusters=clusters, test_clusters=clusters,
                                 cluster_ci_expansion=False)
    assert result.ci_expanded is None and result.expansion_df is None
    with pytest.raises(TypeError, match="cluster_ci_expansion"):
        TwoGroupsEffectSize(control, control + 1, "mean_diff", resamples=200, control_clusters=clusters,
                            test_clusters=clusters, cluster_ci_expansion="yes")
    with pytest.raises(TypeError, match="cluster_ci_expansion"):
        load(DF, cluster_col="ID", cluster_ci_expansion=1, **PAIRED_KWARGS)
    with pytest.raises(ValueError, match="same column as `x`"):
        load(DF, cluster_col="Level", **PAIRED_KWARGS)


def test_expansion_for_every_effect_size_and_plot(clustered):
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")
    for effect_size in ("mean_diff", "median_diff", "cohens_d", "hedges_g"):
        results = getattr(clustered, effect_size).results
        assert (results["ci_expanded"] > 95).all(), effect_size
        assert (results["bca_low"] <= results["difference"]).all()
        assert (results["difference"] <= results["bca_high"]).all()

    unpaired = load(DF, idx=("L1", "L3"), x="Level", y="Y", cluster_col="ID", resamples=300)
    assert unpaired.cliffs_delta.results["ci_expanded"].iloc[0] > 95

    df = DF.copy()
    df["Yes"] = (np.random.default_rng(1).random(len(df)) < 0.4).astype(int)
    proportional = load(df, idx=("L1", "L3"), x="Level", y="Yes", proportional=True,
                        cluster_col="ID", resamples=300)
    for effect_size in ("mean_diff", "cohens_h"):
        assert getattr(proportional, effect_size).results["ci_expanded"].iloc[0] > 95

    assert clustered.mean_diff.plot(show_baseline_ec=True) is not None
    plt.close("all")


def test_unexpanded_limits_are_reported_alongside_the_expanded_ones(clustered, naive):
    expanded = clustered.mean_diff.results
    unexpanded = load(DF, cluster_col="ID", cluster_ci_expansion=False, **PAIRED_KWARGS).mean_diff.results

    # The unexpanded limits are the limits the interval would have without
    # expansion, and lie inside the expanded interval.
    for kind in ("bca", "pct", "bec_bca", "bec_pct"):
        for side in ("low", "high"):
            assert expanded[f"{kind}_{side}_unexpanded"].to_numpy() == pytest.approx(
                unexpanded[f"{kind}_{side}"].to_numpy()), (kind, side)
        assert (expanded[f"{kind}_low"] <= expanded[f"{kind}_low_unexpanded"]).all()
        assert (expanded[f"{kind}_high_unexpanded"] <= expanded[f"{kind}_high"]).all()

    # The same values are available on the effect size object.
    control, test = DF["Level"] == "L1", DF["Level"] == "L2"
    two_groups = TwoGroupsEffectSize(
        DF.loc[control, "Y"].to_numpy(), DF.loc[test, "Y"].to_numpy(), "mean_diff",
        is_paired="sequential", resamples=PAIRED_KWARGS["resamples"], permutation_count=10,
        control_clusters=DF.loc[control, "ID"].to_numpy(), test_clusters=DF.loc[test, "ID"].to_numpy())
    assert two_groups.bca_low <= two_groups.bca_low_unexpanded < two_groups.bca_high_unexpanded <= two_groups.bca_high
    assert two_groups.pct_low <= two_groups.pct_low_unexpanded < two_groups.pct_high_unexpanded <= two_groups.pct_high

    # Nothing to report when the interval is not expanded.
    for results in (unexpanded, naive.mean_diff.results):
        assert not any("unexpanded" in c for c in results.columns)

    # Delta-delta and mini-meta.
    df = DF[DF["Level"].isin(["L1", "L3"])].copy()
    df["Env"] = np.where(df["pair"] % 4 < 2, "A", "B")
    delta2_kwargs = dict(x=["Level", "Env"], y="Y", delta2=True, experiment="Env", paired="sequential",
                         id_col="pair", cluster_col="ID", resamples=500)
    mini_meta = load(df, idx=(("L1", "L3"),), x="Level", y="Y", mini_meta=True, paired="sequential",
                     id_col="pair", cluster_col="ID", resamples=500)
    for obj in (load(df, **delta2_kwargs).mean_diff.delta_delta, mini_meta.mean_diff.mini_meta):
        assert obj.bca_low <= obj.bca_low_unexpanded < obj.bca_high_unexpanded <= obj.bca_high
        assert obj.pct_low <= obj.pct_low_unexpanded < obj.pct_high_unexpanded <= obj.pct_high
        assert {"bca_low_unexpanded", "pct_high_unexpanded"} <= set(obj.results.columns)
    off = load(df, cluster_ci_expansion=False, **delta2_kwargs).mean_diff.delta_delta
    assert off.bca_low_unexpanded is None
    assert not any("unexpanded" in c for c in off.results.columns)


def _ci_segments(ax, horizontal=False):
    """(position, low, high, linewidth) of every straight line drawn along the effect size axis."""
    segments = []
    for line in ax.get_lines():
        x = np.asarray(line.get_xdata(), dtype=float)
        y = np.asarray(line.get_ydata(), dtype=float)
        along, across = (y, x) if horizontal else (x, y)
        if len(along) == 2 and along[0] == along[1] and across[0] != across[1]:
            segments.append((along[0], across.min(), across.max(), line.get_linewidth()))
    return sorted(segments)


def test_plots_draw_the_expansion_as_a_thinner_line(clustered, naive):
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")
    results = clustered.mean_diff.results

    # At each contrast a thin line spans the expanded interval, and a thick bar
    # the unexpanded interval, which the bootstrap distribution spans.
    fig = clustered.mean_diff.plot()
    segments = _ci_segments(fig.axes[1])
    for j, row in enumerate(results.itertuples(), start=1):
        at_j = [s for s in segments if s[0] == j]
        assert len(at_j) == 2
        thin, thick = sorted(at_j, key=lambda s: s[3])
        assert thin[3] < thick[3]
        assert (thin[1], thin[2]) == pytest.approx((row.bca_low, row.bca_high))
        assert (thick[1], thick[2]) == pytest.approx((row.bca_low_unexpanded, row.bca_high_unexpanded))
    plt.close("all")

    # The percentile interval, horizontal plots and custom styling of the thin line.
    fig = clustered.mean_diff.plot(ci_type="pct", horizontal=True,
                                   contrast_expanded_errorbar_kwargs={"color": "red", "lw": 1.5})
    ax = fig.axes[1]
    thin = [line for line in ax.get_lines() if line.get_color() == "red"]
    assert thin and all(line.get_linewidth() == 1.5 for line in thin)
    thin = sorted(s for s in _ci_segments(ax, horizontal=True) if s[3] == 1.5)
    assert [(lo, hi) for _, lo, hi, _ in thin] == pytest.approx(list(zip(results.pct_low, results.pct_high)))
    plt.close("all")

    # One whisker per contrast when the interval is not expanded.
    opted_out = load(DF, cluster_col="ID", cluster_ci_expansion=False, **PAIRED_KWARGS)
    for obj in (naive, opted_out):
        segments = _ci_segments(obj.mean_diff.plot().axes[1])
        assert len({s[3] for s in segments}) == 1
        assert sorted(s[0] for s in segments) == [1.0, 2.0]
        plt.close("all")

    # The baseline error curve and delta-delta are drawn the same way.
    fig = clustered.mean_diff.plot(show_baseline_ec=True)
    assert len([s for s in _ci_segments(fig.axes[1]) if s[0] == 0]) == 2
    plt.close("all")
    df = DF[DF["Level"].isin(["L1", "L3"])].copy()
    df["Env"] = np.where(df["pair"] % 4 < 2, "A", "B")
    delta2 = load(df, x=["Level", "Env"], y="Y", delta2=True, experiment="Env", paired="sequential",
                  id_col="pair", cluster_col="ID", resamples=500)
    dd = delta2.mean_diff.delta_delta
    drawn = [(lo, hi) for _, lo, hi, _ in _ci_segments(delta2.mean_diff.plot().axes[1])]
    assert any((lo, hi) == pytest.approx((dd.bca_low, dd.bca_high)) for lo, hi in drawn)
    assert any((lo, hi) == pytest.approx((dd.bca_low_unexpanded, dd.bca_high_unexpanded)) for lo, hi in drawn)
    plt.close("all")


def test_forest_plot_draws_the_expansion_as_a_thinner_line(clustered, naive):
    import matplotlib
    import matplotlib.pyplot as plt
    from dabest import forest_plot

    matplotlib.use("Agg")
    results = clustered.mean_diff.results.iloc[0]
    for horizontal in (False, True):
        fig = forest_plot([clustered, naive], idx=[[0], [0]], labels=["clustered", "naive"], horizontal=horizontal)
        segments = _ci_segments(fig.axes[0], horizontal=horizontal)
        at_1 = sorted((s for s in segments if s[0] == 1), key=lambda s: s[3])
        assert len(at_1) == 2
        assert (at_1[0][1], at_1[0][2]) == pytest.approx((results.bca_low, results.bca_high))
        assert (at_1[1][1], at_1[1][2]) == pytest.approx((results.bca_low_unexpanded, results.bca_high_unexpanded))
        assert len([s for s in segments if s[0] == 2]) == 1
        plt.close("all")
