#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Unit tests for Cutana size-stratified unlabeled sampling.

These exercise ``CutanaSource._sample_by_tiles`` directly (no FITS streaming):
the stratification logic lives entirely in catalogue-row selection, so it can
be validated against synthetic catalogues with a controlled size distribution.
"""

import numpy as np
import pandas as pd
import pytest

import anomaly_match as am
from anomaly_match.datasets.cutana_source import (
    CutanaSource,
    _flat_plateau_supplied,
    _flat_size_cap,
    _quota_sample_indices,
    _resolve_size_column,
    _size_bin_edges,
)
from anomaly_match.utils.validate_config import validate_config


def _write_catalogue(path, source_ids, tiles, diameters, *, size_col="diameter_pixel"):
    """Write a minimal Cutana catalogue CSV with a size column.

    Args:
        path: Destination CSV path.
        source_ids: Iterable of source id strings.
        tiles: Iterable of ``fits_file_paths`` tile keys (aligned to ids).
        diameters: Iterable of source sizes (aligned to ids); ``None`` omits
            the size column entirely.
        size_col: Which size column name to use when *diameters* is given.
    """
    data = {
        "SourceID": list(source_ids),
        "RA": np.zeros(len(source_ids)),
        "Dec": np.zeros(len(source_ids)),
        "fits_file_paths": list(tiles),
    }
    if diameters is not None:
        data[size_col] = list(diameters)
    pd.DataFrame(data).to_csv(path, index=False)


def _skewed_catalogue(tmp_path, *, n_small=4500, n_large=500, seed=0):
    """One-tile catalogue with a heavy small-source bias.

    Returns the CutanaSource and the full diameter array of the population so a
    test can compare sampled distributions against it.
    """
    rng = np.random.RandomState(seed)
    small = rng.randint(5, 20, n_small)
    large = rng.randint(20, 200, n_large)
    diameters = np.concatenate([small, large])
    n = len(diameters)
    cat = tmp_path / "skewed.csv"
    _write_catalogue(
        cat,
        source_ids=[f"s{i}" for i in range(n)],
        tiles=["['tileA.fits']"] * n,
        diameters=diameters,
    )
    cfg = am.get_default_cfg()
    cfg.data_dir = str(tmp_path)
    return CutanaSource(cfg, lazy=False), diameters


class TestSizeBinEdges:
    """Log bins: fixed width up to max_px, continued at that width over the tail."""

    def test_flattened_region_spans_min_to_max_px_in_n_bins(self):
        # No sources above max_px → edges are exactly [min, max_px] in n_bins.
        sizes = np.array([5.0, 12.0, 40.0, 150.0])
        edges = _size_bin_edges(sizes, n_bins=20, max_px=200)
        assert edges is not None
        assert len(edges) == 21
        assert edges[0] == pytest.approx(5.0)
        assert edges[-1] == pytest.approx(200.0)  # explicit px, not a data percentile
        assert np.all(np.diff(edges) > 0)  # monotonic

    def test_edges_continue_past_max_px_to_cover_tail(self):
        # A giant above max_px gets its own bins rather than being folded into one
        # top bin — this is what lets the sampler keep it and the UI draw the tail.
        sizes = np.array([5.0, 12.0, 40.0, 2000.0])
        edges = _size_bin_edges(sizes, n_bins=20, max_px=200)
        assert edges is not None
        assert len(edges) > 21  # extra bins beyond the 20 flattened-region bins
        assert edges[-1] >= 2000.0  # covers the largest source
        assert np.isclose(edges, 200.0).any()  # max_px stays a bin boundary

    def test_ignores_nonpositive_and_nonfinite_for_lower_edge(self):
        sizes = np.array([np.nan, 0.0, -3.0, 10.0, 30.0])
        edges = _size_bin_edges(sizes, n_bins=10, max_px=200)
        assert edges is not None
        assert edges[0] == pytest.approx(10.0)  # smallest positive finite size

    def test_none_when_no_positive_sizes(self):
        assert _size_bin_edges(np.array([np.nan, 0.0, -1.0]), n_bins=10, max_px=200) is None

    def test_none_when_min_exceeds_max_px(self):
        # Every source larger than the window → nothing to flatten below max_px.
        assert _size_bin_edges(np.full(5, 300.0), n_bins=10, max_px=200) is None


class TestFlatSizeCap:
    """Water-filling cap: level the dense bins, leave the sparse ones intact."""

    def test_levels_dense_bins_and_keeps_sparse(self):
        counts = np.array([1000, 1000, 10, 5, 1])
        cap = _flat_size_cap(counts, take=1000)
        # Largest cap that fits the budget: cap fits, cap+1 overflows.
        assert int(np.minimum(counts, cap).sum()) <= 1000
        assert int(np.minimum(counts, cap + 1).sum()) > 1000
        # Sparse bins sit below the cap, so they are kept whole (not levelled away).
        assert cap >= 10

    def test_returns_max_when_everything_fits(self):
        counts = np.array([3, 5, 2])
        assert _flat_size_cap(counts, take=100) == 5


class TestFlatPlateauSupplied:
    """The top flattened bin's supply gates whether more tiles are pooled."""

    def test_true_when_top_bin_well_populated(self):
        # Plenty of sources right below max_px → the top flattened bin fills.
        sizes = np.concatenate([np.full(500, 10.0), np.full(200, 195.0)])
        assert _flat_plateau_supplied(sizes, n_bins=30, max_px=200, min_per_bin=50)

    def test_false_when_top_bin_starved(self):
        # A large-source-poor pool: nothing near max_px → top bin can't fill.
        sizes = np.full(5000, 12.0)
        assert not _flat_plateau_supplied(sizes, n_bins=30, max_px=200, min_per_bin=50)

    def test_true_when_nothing_below_max_px_to_flatten(self):
        # No sub-max_px sources → adding tiles can't help; don't loop forever.
        assert _flat_plateau_supplied(np.full(10, 500.0), n_bins=30, max_px=200, min_per_bin=50)


class TestQuotaSampleIndices:
    """Water-filled draw: uniform in log size (capped at availability), robust to sparsity."""

    def test_is_uniform_not_distribution_shaped_when_dense_above_max_px(self):
        # Intent: sample UNIFORMLY in size, not reproduce the population. If a
        # dense region sits *above* max_px, those bins must be levelled to the
        # same plateau height as the rest (flat) — NOT subsampled in proportion to
        # their population (which would reproduce the distribution's shape there).
        rng = np.random.RandomState(7)
        # Bulk (~100px) is above max_px=90, so the bins above max_px are dense.
        sizes = np.clip(rng.normal(100, 20, 100000), 0, 200)
        max_px, n_bins = 90.0, 12
        idx = _quota_sample_indices(sizes, take=1000, n_bins=n_bins, max_px=max_px, rng=rng)
        edges = _size_bin_edges(sizes, n_bins, max_px)
        drawn_bins = np.clip(np.digitize(sizes[idx], edges[1:-1]), 0, len(edges) - 2)
        out = np.bincount(drawn_bins, minlength=len(edges) - 1)
        # The dense bins (population well above the plateau height) that carry the
        # draw all sit at roughly the same height — a flat plateau, not a taper
        # that follows the Gaussian's declining density above its mode.
        dense = out[out > out.max() // 2]
        assert dense.max() - dense.min() <= 2  # essentially flat
        # A distribution-shaped draw would instead make bins above the mode fall
        # off in proportion to population; assert we did NOT do that.
        pop = np.bincount(
            np.clip(np.digitize(sizes, edges[1:-1]), 0, len(edges) - 2), minlength=len(edges) - 1
        )
        busiest = int(np.argmax(pop))
        assert out[busiest] < pop[busiest] // 100  # heavily levelled, not proportional

    def test_flattens_small_bulk_and_boosts_large(self):
        rng = np.random.RandomState(0)
        # 9000 tiny (~10px) + 1000 spread across 20-200px — ~10% large by count.
        sizes = np.concatenate(
            [rng.randint(5, 15, 9000).astype(float), rng.randint(20, 200, 1000).astype(float)]
        )
        idx = _quota_sample_indices(sizes, take=2000, n_bins=30, max_px=200, rng=rng)
        assert len(idx) == 2000
        assert len(np.unique(idx)) == len(idx)  # without replacement
        # The rare large end is massively boosted over its ~10% population share.
        assert np.mean(sizes[idx] >= 20) > 0.35

    def test_keeps_large_tail_without_lumping_or_dropping(self):
        # Sources span well past max_px. Each distinct large size must survive in
        # the draw (its own bin), not be folded into one top bin or dropped — the
        # behaviour this change fixed.
        rng = np.random.RandomState(5)
        small = rng.randint(5, 30, 9000).astype(float)
        tail = np.array([250.0, 400.0, 700.0, 1200.0, 2000.0])  # all above max_px=200
        sizes = np.concatenate([small, np.repeat(tail, 50)])
        idx = _quota_sample_indices(sizes, take=3000, n_bins=30, max_px=200, rng=rng)
        drawn = sizes[idx]
        assert len(np.unique(idx)) == len(idx)
        for large in tail:
            assert np.any(drawn == large), f"lost large source {large}px"
        assert drawn.max() == 2000.0  # the biggest giant is present

    def test_never_trips_replace_guard_on_sparse_large_bins(self):
        # Large bins here hold ~1 source each — far below take/n_bins. A *weighted*
        # without-replacement draw trips pandas' "size * max_weight > 1" guard on
        # exactly this shape; the quota draw must complete without error.
        rng = np.random.RandomState(1)
        sizes = np.concatenate([np.full(5000, 8.0), np.linspace(20, 200, 40)])
        idx = _quota_sample_indices(sizes, take=1000, n_bins=30, max_px=200, rng=rng)
        assert len(idx) == 1000
        assert len(np.unique(idx)) == 1000

    def test_take_exceeding_population_returns_all(self):
        rng = np.random.RandomState(2)
        sizes = np.array([5.0, 10.0, 50.0, 150.0])
        idx = _quota_sample_indices(sizes, take=100, n_bins=30, max_px=200, rng=rng)
        assert sorted(idx.tolist()) == [0, 1, 2, 3]

    def test_falls_back_to_uniform_when_all_sizes_above_max_px(self):
        # No sub-max_px sources → no bins to flatten → plain without-replacement draw.
        rng = np.random.RandomState(3)
        idx = _quota_sample_indices(np.full(50, 500.0), take=10, n_bins=30, max_px=200, rng=rng)
        assert len(idx) == 10
        assert len(np.unique(idx)) == 10

    def test_nonfinite_sizes_do_not_crash(self):
        # NaN sizes can't be binned; they stay eligible only for the top-up.
        rng = np.random.RandomState(4)
        sizes = np.concatenate([np.full(100, 10.0), np.full(100, 50.0), np.full(5, np.nan)])
        idx = _quota_sample_indices(sizes, take=50, n_bins=10, max_px=200, rng=rng)
        assert len(idx) == 50
        assert len(np.unique(idx)) == 50


class TestResolveSizeColumn:
    """Size-column resolution prefers diameter_pixel over diameter_arcsec."""

    def test_prefers_pixel(self):
        df = pd.DataFrame({"diameter_pixel": [1], "diameter_arcsec": [2]})
        assert _resolve_size_column(df) == "diameter_pixel"

    def test_falls_back_to_arcsec(self):
        df = pd.DataFrame({"diameter_arcsec": [2.0]})
        assert _resolve_size_column(df) == "diameter_arcsec"

    def test_none_when_absent(self):
        assert _resolve_size_column(pd.DataFrame({"SourceID": ["s"]})) is None


class TestSizeStratifiedSampling:
    """Stratified sampling flattens the pool's size distribution."""

    def test_stratified_pool_is_flatter_and_larger_than_uniform(self, tmp_path):
        source, population = _skewed_catalogue(tmp_path)
        rng_u = np.random.RandomState(123)
        rng_s = np.random.RandomState(123)

        n = 800
        uniform = source._sample_by_tiles(n, 8, set(), rng_u, stratify_size=False)
        stratified = source._sample_by_tiles(n, 8, set(), rng_s, stratify_size=True)

        u_sizes = uniform["diameter_pixel"].to_numpy(dtype=float)
        s_sizes = stratified["diameter_pixel"].to_numpy(dtype=float)

        # Stratification pulls in the rare large sources → higher mean size.
        assert s_sizes.mean() > u_sizes.mean() * 1.3

        # Flatter: bin both samples over the population range and compare the
        # coefficient of variation of bin counts (lower = more uniform).
        edges = np.linspace(population.min(), population.max(), 21)
        u_counts = np.histogram(u_sizes, bins=edges)[0]
        s_counts = np.histogram(s_sizes, bins=edges)[0]
        u_cv = u_counts.std() / u_counts.mean()
        s_cv = s_counts.std() / s_counts.mean()
        assert s_cv < u_cv

    def test_heavy_tail_does_not_overshoot(self, tmp_path):
        # Euclid-like: a small-source bulk plus a long heavy tail with a few giant
        # artifacts. Log+clip binning must lift the pool's median moderately
        # without letting the rare giants dominate (the bug linear bins had).
        rng = np.random.RandomState(0)
        bulk = rng.randint(5, 25, 9000)  # tiny majority
        tail = (rng.pareto(1.5, 1000) * 20 + 25).astype(int)  # heavy tail
        giants = np.array([1500, 2000, 2400])  # rare artifacts past p99
        diameters = np.concatenate([bulk, tail, giants])
        cat = tmp_path / "heavy.csv"
        _write_catalogue(
            cat,
            source_ids=[f"s{i}" for i in range(len(diameters))],
            tiles=["['t.fits']"] * len(diameters),
            diameters=diameters,
        )
        cfg = am.get_default_cfg()
        cfg.data_dir = str(tmp_path)
        source = CutanaSource(cfg, lazy=False)

        n = 2000
        uniform = source._sample_by_tiles(
            n, 1, set(), np.random.RandomState(1), stratify_size=False
        )
        strat = source._sample_by_tiles(n, 1, set(), np.random.RandomState(1), stratify_size=True)
        u = uniform["diameter_pixel"].to_numpy(float)
        s = strat["diameter_pixel"].to_numpy(float)

        # Substantially boosts medium/large-source exposure (the whole point)...
        assert (s > 30).mean() > 2 * (u > 30).mean()
        assert (s > 30).mean() > 0.15
        # ...without overshooting into a tail-dominated pool: the median stays in a
        # sane range and the rare giant artifacts don't take over.
        assert np.median(s) < 100
        assert (s > 1000).mean() < 0.02

    def _many_tile_catalogue(self, tmp_path, tile_sizes, *, diam_range=(5, 200), seed=1):
        """Write a multi-tile catalogue; ``tile_sizes`` maps tile name -> row count."""
        rng = np.random.RandomState(seed)
        rows_ids, rows_tiles, rows_diam = [], [], []
        idx = 0
        for tile_name, count in tile_sizes.items():
            for _ in range(count):
                rows_ids.append(f"s{idx}")
                rows_tiles.append(f"['{tile_name}.fits']")
                rows_diam.append(int(rng.randint(*diam_range)))
                idx += 1
        _write_catalogue(tmp_path / "multi.csv", rows_ids, rows_tiles, rows_diam)
        cfg = am.get_default_cfg()
        cfg.data_dir = str(tmp_path)
        return cfg

    def test_tile_selection_is_random_not_population_biased(self, tmp_path):
        # One dominant tile plus several equally-sized others, all size-rich. Random
        # selection must NOT always take the biggest tile — across seeds the pool
        # rotates and the small tiles appear — and it must vary seed to seed. (A
        # population-ranked selector would pin the big tile every time.)
        cfg = self._many_tile_catalogue(tmp_path, {"big": 4000, **{f"t{i}": 800 for i in range(8)}})
        cfg.cutana_max_unlabeled_tiles = 3
        source = CutanaSource(cfg, lazy=False)

        seen_tiles, per_seed = set(), []
        for seed in range(10):
            res = source._sample_by_tiles(
                600, 3, set(), np.random.RandomState(seed), stratify_size=True
            )
            tiles = frozenset(res["fits_file_paths"].unique())
            per_seed.append(tiles)
            seen_tiles |= tiles
        # Selection rotates across seeds (not one frozen set) and reaches beyond the
        # single biggest tile into the smaller ones.
        assert len(set(per_seed)) > 1
        assert len(seen_tiles) > cfg.cutana_max_unlabeled_tiles
        assert any(t != "['big.fits']" for tiles in per_seed for t in tiles)

    def test_pools_more_tiles_when_large_sources_scarce(self, tmp_path):
        # Every tile is large-source-poor (mostly < 30px), so a baseline draw of
        # max_tiles can't fill the top flattened bin → the sampler tops up with more
        # random tiles, exceeding max_tiles (bounded by the pool factor).
        cfg = self._many_tile_catalogue(
            tmp_path, {f"t{i}": 1500 for i in range(12)}, diam_range=(5, 30)
        )
        cfg.cutana_max_unlabeled_tiles = 2
        source = CutanaSource(cfg, lazy=False)

        res = source._sample_by_tiles(6000, 2, set(), np.random.RandomState(0), stratify_size=True)
        n_tiles = res["fits_file_paths"].nunique()
        assert n_tiles > 2  # topped up past the baseline
        assert n_tiles <= 3 * 2  # bounded by _STRATIFY_TILE_POOL_FACTOR × max_tiles

    def test_scan_stops_early_and_bounds_catalogue_reads(self, tmp_path, monkeypatch):
        """The stratified sampler must not read every catalogue: once enough
        tiles are discovered it stops.  This is what keeps it bounded on DR1's
        thousands of catalogue files instead of scanning all of them."""
        # Six catalogues, each with eight size-rich tiles — any single one
        # already exceeds tile_target (= 3 × max_tiles = 6 here).
        for file_index in range(6):
            rng = np.random.RandomState(file_index)
            ids, tiles, diam = [], [], []
            for tile_index in range(8):
                for _ in range(300):
                    ids.append(f"f{file_index}t{tile_index}s{len(ids)}")
                    tiles.append(f"['f{file_index}_t{tile_index}.fits']")
                    diam.append(int(rng.randint(5, 200)))
            _write_catalogue(tmp_path / f"cat_{file_index}.csv", ids, tiles, diam)

        cfg = am.get_default_cfg()
        cfg.data_dir = str(tmp_path)
        cfg.cutana_max_unlabeled_tiles = 2
        source = CutanaSource(cfg, lazy=False)
        assert len(source._catalogue_files) == 6

        reads = {"n": 0}
        original_load = source._load_catalogue

        def counting_load(path):
            reads["n"] += 1
            return original_load(path)

        monkeypatch.setattr(source, "_load_catalogue", counting_load)

        res = source._sample_by_tiles(500, 2, set(), np.random.RandomState(0), stratify_size=True)

        assert not res.empty
        # Each catalogue has 8 tiles and tile_target = 3 x max_tiles = 6, so the
        # first file read already covers the target — exactly one catalogue is
        # scanned regardless of permutation order.
        assert reads["n"] == 1

    def test_missing_size_column_raises_when_stratifying(self, tmp_path):
        cat = tmp_path / "no_size.csv"
        _write_catalogue(
            cat,
            source_ids=[f"s{i}" for i in range(50)],
            tiles=["['t.fits']"] * 50,
            diameters=None,
        )
        cfg = am.get_default_cfg()
        cfg.data_dir = str(tmp_path)
        source = CutanaSource(cfg, lazy=False)
        with pytest.raises(ValueError, match="no source-size column"):
            source._sample_by_tiles(20, 8, set(), np.random.RandomState(0), stratify_size=True)

    def test_flag_off_works_without_size_column(self, tmp_path):
        # The default (random) path must not require a diameter column.
        cat = tmp_path / "no_size.csv"
        _write_catalogue(
            cat,
            source_ids=[f"s{i}" for i in range(200)],
            tiles=["['t.fits']"] * 200,
            diameters=None,
        )
        cfg = am.get_default_cfg()
        cfg.data_dir = str(tmp_path)
        source = CutanaSource(cfg, lazy=False)
        # n=50, max_tiles=1 → per_tile_cap=50 draws 50 of the single tile's 200 rows.
        result = source._sample_by_tiles(
            50, 1, set(), np.random.RandomState(0), stratify_size=False
        )
        assert len(result) == 50

    def test_excluded_ids_are_dropped(self, tmp_path):
        source, population = _skewed_catalogue(tmp_path)
        exclude = {f"s{i}" for i in range(100)}
        result = source._sample_by_tiles(
            500, 8, exclude, np.random.RandomState(0), stratify_size=True
        )
        assert exclude.isdisjoint(set(result["SourceID"].astype(str)))


class TestLastUnlabeledSizes:
    """The source records the diameters of its most recent unlabeled batch so
    the training subprocess can emit a size histogram (PR3)."""

    def test_sizes_recorded_after_unlabeled_sample(self, tmp_path):
        source, _population = _skewed_catalogue(tmp_path)
        assert source.get_last_unlabeled_sizes() is None  # nothing sampled yet
        merged = source._sample_by_tiles(300, 1, set(), np.random.RandomState(0))
        source._record_unlabeled_sizes(merged)
        sizes = source.get_last_unlabeled_sizes()
        assert sizes is not None
        assert len(sizes) == len(merged)
        assert np.isfinite(sizes).all()

    def test_population_sizes_recorded_by_sampler(self, tmp_path):
        # The sampler records the candidate population of the drawn tiles (the
        # baseline the UI overlays the sample against), covering every source in
        # the single tile — a superset of the smaller sampled pool.
        source, population = _skewed_catalogue(tmp_path)
        assert source.get_last_unlabeled_population_sizes() is None  # nothing sampled yet
        source._sample_by_tiles(300, 1, set(), np.random.RandomState(0))
        pop_sizes = source.get_last_unlabeled_population_sizes()
        assert pop_sizes is not None
        # Single-tile catalogue → the population is the whole catalogue.
        assert len(pop_sizes) == len(population)
        assert np.isfinite(pop_sizes).all()

    def test_population_sizes_recorded_when_stratified(self, tmp_path):
        source, population = _skewed_catalogue(tmp_path)
        source._sample_by_tiles(300, 8, set(), np.random.RandomState(0), stratify_size=True)
        pop_sizes = source.get_last_unlabeled_population_sizes()
        assert pop_sizes is not None
        assert len(pop_sizes) == len(population)

    def test_population_sizes_none_without_size_column(self, tmp_path):
        cat = tmp_path / "no_size.csv"
        _write_catalogue(
            cat,
            source_ids=[f"s{i}" for i in range(20)],
            tiles=["['t.fits']"] * 20,
            diameters=None,
        )
        cfg = am.get_default_cfg()
        cfg.data_dir = str(tmp_path)
        source = CutanaSource(cfg, lazy=False)
        source._sample_by_tiles(10, 1, set(), np.random.RandomState(0))
        assert source.get_last_unlabeled_population_sizes() is None

    def test_sizes_none_without_size_column(self, tmp_path):
        cat = tmp_path / "no_size.csv"
        _write_catalogue(
            cat,
            source_ids=[f"s{i}" for i in range(20)],
            tiles=["['t.fits']"] * 20,
            diameters=None,
        )
        cfg = am.get_default_cfg()
        cfg.data_dir = str(tmp_path)
        source = CutanaSource(cfg, lazy=False)
        merged = source._sample_by_tiles(10, 1, set(), np.random.RandomState(0))
        source._record_unlabeled_sizes(merged)
        assert source.get_last_unlabeled_sizes() is None

    def test_base_source_returns_none(self):
        # Image/Zarr sources expose no per-source size.
        from anomaly_match.datasets.training_data_source import ImageFolderSource

        cfg = am.get_default_cfg()
        cfg.data_dir = "tests/test_data/grayscale"
        src = ImageFolderSource(cfg)
        assert src.get_last_unlabeled_sizes() is None

    def test_callback_fires_with_sizes_available_before_streaming(self, tmp_path):
        # The subprocess wires this hook to emit the size histogram early; it must
        # fire from _record_unlabeled_sizes (which runs before _stream_cutouts) and
        # the recorded sizes must already be readable when it does.
        source, _ = _skewed_catalogue(tmp_path)
        seen = {}

        source.unlabeled_sizes_callback = lambda: seen.setdefault(
            "sizes", source.get_last_unlabeled_sizes()
        )
        merged = source._sample_by_tiles(200, 1, set(), np.random.RandomState(0))
        source._record_unlabeled_sizes(merged)

        assert "sizes" in seen  # callback fired
        assert seen["sizes"] is not None and len(seen["sizes"]) > 0  # sizes ready when it fired


class TestSamplingLogs:
    """Sampling states in the log whether stratification was applied and how."""

    def _capture(self, fn):
        from loguru import logger

        messages: list[str] = []
        sink_id = logger.add(messages.append, level="INFO", format="{message}")
        try:
            fn()
        finally:
            logger.remove(sink_id)
        return "\n".join(messages)

    def test_logs_on_mode_when_stratifying(self, tmp_path):
        source, _ = _skewed_catalogue(tmp_path)
        source._cfg.cutana_stratify_source_size = True
        out = self._capture(
            lambda: source._record_unlabeled_sizes(
                source._sample_by_tiles(300, 1, set(), np.random.RandomState(0), stratify_size=True)
            )
        )
        assert "ON (uniform-in-size)" in out
        assert "median" in out.lower()

    def test_logs_off_mode_when_not_stratifying(self, tmp_path):
        source, _ = _skewed_catalogue(tmp_path)
        source._cfg.cutana_stratify_source_size = False
        out = self._capture(
            lambda: source._record_unlabeled_sizes(
                source._sample_by_tiles(300, 1, set(), np.random.RandomState(0))
            )
        )
        assert "OFF (random)" in out

    def test_stratified_method_logs_tile_selection(self, tmp_path):
        source, _ = _skewed_catalogue(tmp_path)
        out = self._capture(
            lambda: source._sample_by_tiles_size_stratified(200, 8, set(), np.random.RandomState(0))
        )
        assert "Size stratification" in out
        assert "tiles" in out.lower()


class TestUnlabeledSeed:
    """The unlabelled RNG seed varies with the labelled set so retrain cycles
    draw a fresh pool, yet stays reproducible for a given label set."""

    def test_empty_exclude_uses_base_seed(self, tmp_path):
        source, _ = _skewed_catalogue(tmp_path)
        assert source._unlabeled_seed(set()) == int(source._cfg.seed)

    def test_different_labels_give_different_seed(self, tmp_path):
        source, _ = _skewed_catalogue(tmp_path)
        assert source._unlabeled_seed({"a", "b"}) != source._unlabeled_seed({"a", "c"})

    def test_seed_is_order_independent_and_stable(self, tmp_path):
        source, _ = _skewed_catalogue(tmp_path)
        # Set insertion order must not matter, and it must be reproducible.
        assert source._unlabeled_seed({"a", "b", "c"}) == source._unlabeled_seed({"c", "a", "b"})

    def test_fresh_pool_across_cycles(self, tmp_path):
        # Two cycles with different labelled sets must sample different pools;
        # the same labelled set must reproduce the same pool.
        source, _ = _skewed_catalogue(tmp_path)

        def _pool(exclude):
            rng = np.random.RandomState(source._unlabeled_seed(exclude))
            df = source._sample_by_tiles(500, 1, set(str(x) for x in exclude), rng)
            return set(df["SourceID"].astype(str))

        cycle1 = _pool({"s0"})
        cycle2 = _pool({"s0", "s1", "s2"})
        assert cycle1 != cycle2
        assert _pool({"s0"}) == cycle1  # reproducible for the same label set


class TestConfigRegistration:
    """The new flag is registered and defaults off."""

    def test_default_is_false(self):
        assert am.get_default_cfg().cutana_stratify_source_size is False

    def test_default_cfg_validates(self):
        # validate_config requires the key to be present and boolean.
        cfg = am.get_default_cfg()
        cfg.data_dir = "."
        validate_config(cfg)

    def test_non_bool_value_rejected(self):
        cfg = am.get_default_cfg()
        cfg.data_dir = "."
        cfg.cutana_stratify_source_size = "yes"
        with pytest.raises(ValueError, match="cutana_stratify_source_size"):
            validate_config(cfg)
