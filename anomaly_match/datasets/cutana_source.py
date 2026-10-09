#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

"""Cutana catalogue data source for training from streaming cutouts."""

from __future__ import annotations

import hashlib
import inspect
import os
import shutil
import tempfile
from pathlib import Path

import cutana
import numpy as np
import pandas as pd
from cutana import StreamingOrchestrator
from cutana.catalogue_preprocessor import extract_filter_name, parse_fits_file_paths
from dotmap import DotMap
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod
from loguru import logger
from tqdm import tqdm

from anomaly_match.data_io.container_loaders import apply_channel_combination_to_cutana_batch
from anomaly_match.data_io.load_images import _get_channel_combination_array, _resize_param_list
from anomaly_match.datasets.training_data_source import TrainingDataSource
from anomaly_match.utils.tqdm_logging import tqdm_logging_file

# Catalogue columns Cutana accepts for a source's size, in preference order.
# Cutana requires one of these (catalogue_preprocessor validates it at
# extraction), so size stratification can always rely on one being present.
_SIZE_COLUMNS = ("diameter_pixel", "diameter_arcsec")

# Size-stratified sampling draws tiles at random (not by population, which would
# bias the pool toward the densest fields). If a random draw of ``max_tiles`` is
# short on large sources, up to this many × ``max_tiles`` tiles are pooled to
# refill the large-size bins — bounding the extra Cutana streaming cost.
_STRATIFY_TILE_POOL_FACTOR = 3


def _resolve_size_column(df: pd.DataFrame) -> str | None:
    """Return the catalogue's source-size column, or ``None`` if absent.

    Args:
        df: A catalogue DataFrame.

    Returns:
        The first of :data:`_SIZE_COLUMNS` present in *df*, else ``None``.
    """
    for col in _SIZE_COLUMNS:
        if col in df.columns:
            return col
    return None


def _finite_sizes(df: pd.DataFrame) -> np.ndarray | None:
    """Return the finite source diameters of *df*, or ``None`` if unsized.

    Args:
        df: A catalogue DataFrame.

    Returns:
        Finite (NaN/inf-dropped) diameters as a float array, or ``None`` when the
        catalogue carries no diameter column.
    """
    col = _resolve_size_column(df)
    if col is None:
        return None
    sizes = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    return sizes[np.isfinite(sizes)]


def _size_bin_edges(sizes: np.ndarray, n_bins: int, max_px: float) -> np.ndarray | None:
    """Log-spaced size-bin edges covering the full size range, with fixed width.

    The bin *width* (in dex) is set so ``[min_positive, max_px]`` spans exactly
    ``n_bins`` bins — that fixes the resolution of the flattened region to an
    explicit, physically meaningful window (e.g. 200px) regardless of
    catalogue-to-catalogue tail drift. Edges then continue at the **same width**
    past ``max_px`` up to the largest source, so the heavy tail (Euclid has a
    handful of sources at 1000s of px) gets its own bins rather than being folded
    into one lump — the sampler keeps every large source and the UI histogram can
    show the full size range up to the largest source instead of a cliff at
    ``max_px``.

    The sampler and the UI size histogram share these edges so a drawn pool and
    its plotted distribution line up.

    Args:
        sizes: Source sizes; may contain ``NaN``/``inf``/non-positive values.
        n_bins: Number of bins spanning ``[min_positive, max_px]``.
        max_px: Upper edge of the flattened region, in pixels; bins continue past
            it at the same width to cover the tail.

    Returns:
        Monotonic log-spaced edges (``>= n_bins + 1`` of them), or ``None`` when
        ill-defined (no positive sizes, or ``min >= max_px``) and the caller
        should fall back to uniform sampling.
    """
    finite_positive = sizes[np.isfinite(sizes) & (sizes > 0)]
    if finite_positive.size == 0:
        return None
    lo = float(finite_positive.min())
    if lo >= max_px:
        return None
    hi = float(finite_positive.max())
    step = (np.log10(max_px) - np.log10(lo)) / n_bins  # dex per flattened-region bin
    n_total = n_bins if hi <= max_px else int(np.ceil((np.log10(hi) - np.log10(lo)) / step))
    return lo * 10.0 ** (np.arange(n_total + 1) * step)


def _flat_plateau_supplied(sizes: np.ndarray, n_bins: int, max_px: float, min_per_bin: int) -> bool:
    """Whether the top flattened bin holds enough sources to fill its quota.

    The flat plateau only reaches ``max_px`` if its **sparsest** bin — the top one
    just below ``max_px`` — can supply the per-bin quota. Sizes bin into the same
    fixed-width log bins the sampler uses, and the count in ``[edges[n_bins-1],
    max_px]`` is compared against ``min_per_bin``. The stratified sampler uses this
    to decide whether a random tile draw needs topping up with more tiles.

    Args:
        sizes: Pooled candidate source diameters (may contain ``NaN``/``inf``).
        n_bins: Number of bins spanning ``[min, max_px]``.
        max_px: Top of the flattened region, in pixels.
        min_per_bin: Target sources per flattened bin (the nominal quota).

    Returns:
        ``True`` if the top flattened bin holds enough sources, or if there is no
        flattenable region below ``max_px`` (every source ``>= max_px``, or none
        positive) — neither is a large-source shortfall that more tiles would fix.
    """
    edges = _size_bin_edges(sizes, n_bins, max_px)
    if edges is None:
        return True
    top_bin_low = float(edges[n_bins - 1])
    return int(np.count_nonzero((sizes >= top_bin_low) & (sizes <= max_px))) >= min_per_bin


def _flat_size_cap(counts: np.ndarray, take: int) -> int:
    """Largest per-bin cap ``Q`` with ``sum(min(Q, counts)) <= take``.

    This is the water-filling level that flattens the dense (small-source) bins
    to a common height while leaving the sparse (large-source) bins untouched:
    bins below the cap keep all their sources, bins above it are levelled to
    ``Q``. Solved by bisection on ``Q``.

    Args:
        counts: Per-bin source counts.
        take: Total draw budget.

    Returns:
        The flat cap ``Q`` (0 if every bin must be levelled below one source).
    """
    if counts.sum() <= take:
        return int(counts.max())
    low, high = 0, int(counts.max())
    while low < high:
        mid = (low + high + 1) // 2
        if int(np.minimum(counts, mid).sum()) <= take:
            low = mid
        else:
            high = mid - 1
    return low


def _quota_sample_indices(
    sizes: np.ndarray, take: int, n_bins: int, max_px: float, rng: np.random.RandomState
) -> np.ndarray:
    """Positional indices of a size-flattened, without-replacement draw of *take*.

    The goal is a pool that is **uniform in log source size**, not one that
    reproduces the (heavily small-source-biased) catalogue distribution. Sources
    are binned in log size (see :func:`_size_bin_edges`) and each bin contributes
    ``min(Q, available)`` draws **without replacement**, where ``Q`` is the
    water-filling cap (:func:`_flat_size_cap`) that spends the whole budget. So
    every bin is levelled to the *same* height ``Q`` wherever the catalogue can
    supply it — a flat, uniform-in-size plateau — and any bin with fewer than
    ``Q`` sources simply contributes **all** of them.

    This is why the drawn pool thins out at the large-size end: the rare
    large-source bins hold fewer than ``Q`` sources, so we take **every** one of
    them (none discarded or lumped into a top bin), and the pool tails off there
    only because the catalogue runs out of large sources — *not* because the draw
    is shaped to the population. Above the size where bins can no longer fill
    ``Q``, representation is capped by availability, still never exceeding the
    uniform plateau height.

    A per-bin cap can never trip numpy/pandas' ``size * max_weight > 1``
    without-replacement guard that inverse-frequency *weighting* hits on sparse
    large bins, so it is robust exactly where weighting crashed. Any rounding
    shortfall is handed to the bins with spare capacity (the dense ones, where a
    few extra draws are invisible against the plateau), then a final uniform
    top-up guarantees the total is ``min(take, len(sizes))``.

    Args:
        sizes: Source diameters; may contain ``NaN``/``inf``.
        take: Number of sources to draw.
        n_bins: Number of log-spaced bins spanning ``[min, max_px]``.
        max_px: Top of the flattened (uniform) size range, in pixels.
        rng: Random state for reproducibility.

    Returns:
        Positional indices into *sizes* of the drawn sources.
    """
    take = min(take, len(sizes))
    if take <= 0:
        return np.empty(0, dtype=int)
    edges = _size_bin_edges(sizes, n_bins, max_px)
    if edges is None:
        # No positive sizes below max_px to flatten (e.g. max_px set below the
        # catalogue's smallest diameter). Fall back to a uniform draw, but warn:
        # a misconfigured max_px would otherwise make stratification a silent no-op.
        logger.warning(
            "Size stratification degraded to a uniform random draw: no positive "
            "source sizes below cutana_size_stratify_max_px={}px to flatten. "
            "Lower max_px or check the catalogue's diameter column.",
            max_px,
        )
        return rng.choice(len(sizes), take, replace=False)
    n_total = len(edges) - 1
    # Non-finite/non-positive sizes can't be binned; leave them at -1 so they're
    # eligible only for the final top-up, never for a per-bin quota.
    bin_index = np.full(len(sizes), -1, dtype=int)
    finite = np.isfinite(sizes) & (sizes > 0)
    bin_index[finite] = np.clip(np.digitize(sizes[finite], edges[1:-1]), 0, n_total - 1)
    counts = np.bincount(bin_index[finite], minlength=n_total)

    cap = _flat_size_cap(counts, take)
    alloc = np.minimum(counts, cap)
    # Spend the rounding shortfall on bins that still have sources to give,
    # densest first — a +1 on the plateau bins is imperceptible.
    shortfall = take - int(alloc.sum())
    if shortfall > 0:
        spare = counts - alloc
        for b in np.argsort(-spare):
            if shortfall == 0 or spare[b] == 0:
                break
            alloc[b] += 1
            shortfall -= 1

    picked: list[np.ndarray] = []
    for b in range(n_total):
        if alloc[b] > 0:
            members = np.flatnonzero(bin_index == b)
            picked.append(rng.choice(members, int(alloc[b]), replace=False))
    chosen = np.concatenate(picked) if picked else np.empty(0, dtype=int)

    if chosen.size < take:
        remaining = np.setdiff1d(np.arange(len(sizes)), chosen, assume_unique=True)
        extra = min(take - chosen.size, remaining.size)
        if extra > 0:
            chosen = np.concatenate([chosen, rng.choice(remaining, extra, replace=False)])
    return chosen


class CutanaSource(TrainingDataSource):
    """Load training data from one or more Cutana source catalogues.

    Supports both single catalogue files and directories containing multiple
    CSV/parquet catalogues.  Catalogues are not merged into memory; only
    per-file row counts are stored for lightweight indexing.

    Creates cutouts on demand using Cutana's streaming orchestrator. For labeled
    images, sub-catalogues are filtered to just the labeled source IDs. For
    unlabeled images, random source IDs are sampled across files using
    cumulative offset logic.

    When ``lazy=True``, the O(N) shuffled-index permutation is skipped.
    Unlabeled sampling uses random index generation instead. Labeled images
    must come from a ``LabeledDataCache``, not from this source.

    Args:
        cfg: Configuration; ``cfg.data_dir`` must point to a catalogue
            directory containing ``.csv`` or ``.parquet`` files, or to a
            single catalogue file.
        lazy: If True, skip O(N) permutation for billion-scale catalogues.
    """

    def __init__(self, cfg: DotMap, *, lazy: bool = False) -> None:
        super().__init__(cfg)
        self._lazy = lazy
        data_path = Path(cfg.data_dir)

        # Discover catalogue files and their row counts
        self._catalogue_files: list[tuple[Path, int]] = self._discover_catalogues(data_path)
        if not self._catalogue_files:
            raise ValueError(f"No catalogue files found in {data_path}")

        self._total = sum(count for _, count in self._catalogue_files)

        # Finite source diameters from the most recent unlabeled sample, kept so
        # the training subprocess can surface a size-distribution histogram of
        # the pool the model will actually see (#size-strat PR3).  None until the
        # first ``get_unlabeled_batch`` call, or when the catalogue has no size
        # column.
        self._last_unlabeled_sizes: np.ndarray | None = None
        self._last_unlabeled_size_col: str | None = None
        # Finite source diameters of *every* candidate source in the tiles the
        # last unlabeled batch was drawn from — the population the sampler chose
        # from.  Overlaying the sampled distribution on this baseline is what
        # shows whether size stratification actually flattened the pool (the
        # population is small-skewed; a stratified sample is ~flat in log size).
        # Set by the samplers before ``_record_unlabeled_sizes`` fires the hook.
        self._last_unlabeled_population_sizes: np.ndarray | None = None

        mode = "lazy" if lazy else "normal"
        files_desc = ", ".join(f"{p.name}({c})" for p, c in self._catalogue_files)
        logger.debug(f"CutanaSource({mode}): {self._total} sources from [{files_desc}]")

    @staticmethod
    def _discover_catalogues(data_path: Path) -> list[tuple[Path, int]]:
        """Discover catalogue files and count rows without full DataFrame load.

        Args:
            data_path: Path to a catalogue file or directory of catalogues.

        Returns:
            List of (file_path, row_count) tuples.
        """
        if data_path.is_file():
            count = _count_catalogue_rows(data_path)
            return [(data_path, count)]

        if data_path.is_dir():
            results = []
            for child in sorted(data_path.iterdir()):
                if child.suffix in (".csv", ".parquet") and _is_cutana_catalogue(child):
                    count = _count_catalogue_rows(child)
                    results.append((child, count))
            return results

        return []

    def _load_catalogue(self, path: Path) -> pd.DataFrame:
        """Load a full catalogue DataFrame from a single file.

        Args:
            path: Path to a CSV or parquet catalogue.

        Returns:
            DataFrame with at least ``SourceID`` and ``fits_file_paths`` columns.

        Raises:
            ValueError: If required columns are missing.
        """
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

        required = {"SourceID", "fits_file_paths"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Catalogue {path} missing required columns: {sorted(missing)}")
        return df

    def _sample_by_tiles(
        self,
        n: int,
        max_tiles: int,
        exclude_str: set[str],
        rng: np.random.RandomState,
        *,
        stratify_size: bool = False,
    ) -> pd.DataFrame:
        """Sample up to *n* sources from at most *max_tiles* FITS tile sets.

        Iterates catalogue files in random order, discovers FITS tiles, and
        adds tiles until we have enough sources or hit the tile cap. This
        ensures all sampled sources cluster on few tiles, so Cutana's
        orchestrator creates few batches (each batch = one subprocess spawn).

        When *stratify_size* is set, delegates to
        :meth:`_sample_by_tiles_size_stratified`, which pools random tiles and
        flattens the size distribution across them.

        Args:
            n: Target number of sources.
            max_tiles: Maximum number of unique FITS tile sets to sample from.
            exclude_str: Source IDs (as strings) to exclude.
            rng: Random state for reproducibility.
            stratify_size: When True, sample with inverse size-frequency weights
                so the pool's diameter distribution is ~uniform.

        Returns:
            Merged DataFrame of sampled catalogue rows.
        """
        if stratify_size:
            return self._sample_by_tiles_size_stratified(n, max_tiles, exclude_str, rng)
        # Cap per tile so sources spread across tiles instead of filling
        # from the first one (tiles typically have 10k-100k sources).
        per_tile_cap = max(1, n // max_tiles)

        # Iterate catalogues in random order, collecting tiles
        file_order = rng.permutation(len(self._catalogue_files))

        collected: list[pd.DataFrame] = []
        # Full source lists of the tiles we draw from — the candidate population
        # the UI overlays the sampled pool against (see _record_population_sizes).
        candidate: list[pd.DataFrame] = []
        seen_tiles: set[str] = set()

        for fi in file_order:
            if len(seen_tiles) >= max_tiles:
                break

            path, _ = self._catalogue_files[fi]
            df = self._load_catalogue(path)

            if exclude_str:
                df = df[~df["SourceID"].astype(str).isin(exclude_str)]

            if "fits_file_paths" not in df.columns or df.empty:
                continue

            # Group by tile, pick new tiles randomly
            tile_groups = dict(list(df.groupby("fits_file_paths")))
            new_tiles = [t for t in tile_groups if t not in seen_tiles]
            if not new_tiles:
                continue

            rng.shuffle(new_tiles)
            tiles_to_add = new_tiles[: max_tiles - len(seen_tiles)]

            for tile in tiles_to_add:
                tile_df = tile_groups[tile]
                candidate.append(tile_df)
                take = min(per_tile_cap, len(tile_df))
                if take < len(tile_df):
                    tile_df = tile_df.sample(n=take, random_state=rng)
                collected.append(tile_df)
                seen_tiles.add(tile)

                if len(seen_tiles) >= max_tiles:
                    break

        self._record_population_sizes(candidate)
        if not collected:
            return pd.DataFrame()

        result = pd.concat(collected, ignore_index=True)
        # Trim to exactly n if we overshot
        if len(result) > n:
            result = result.sample(n=n, random_state=rng)
        return result

    def _sample_by_tiles_size_stratified(
        self,
        n: int,
        max_tiles: int,
        exclude_str: set[str],
        rng: np.random.RandomState,
    ) -> pd.DataFrame:
        """Sample ~*n* sources from *max_tiles* random tiles, flat in size.

        Like :meth:`_sample_by_tiles`, tiles are chosen at **random** (no
        population bias); unlike it, the chosen tiles are pooled and drawn with an
        equal per-bin quota over log-spaced size bins up to
        ``cfg.cutana_size_stratify_max_px``, so the returned pool's diameter
        distribution is ~uniform rather than dominated by the small-source bulk.
        Euclid Q1/DR1 catalogues skew heavily small (median ~12px), so an
        unstratified unlabeled pool is almost entirely tiny sources.

        The rare large-source bins (~100–200px) hold only ~70–90 sources per tile,
        so if a random draw of ``max_tiles`` is short on large sources the sampler
        pools a few more random tiles (up to ``_STRATIFY_TILE_POOL_FACTOR ×
        max_tiles``) until the top flattened bin can fill its quota. Pooling before
        the draw (rather than drawing per tile) lets a sparse bin recruit its quota
        across every chosen tile at once.

        Opt-in via ``cfg.cutana_stratify_source_size``. Rather than reading every
        catalogue, it scans them in **random order** and stops once
        ``_STRATIFY_TILE_POOL_FACTOR × max_tiles`` tiles have been discovered — a
        single Q1 catalogue usually already holds far more than enough, and this
        bound is what keeps the feature safe on DR1's thousands of catalogue
        files. Random order (the rng is seeded from the labelled set) means the
        scanned files — and thus the drawn tiles — rotate across retrain cycles.

        Args:
            n: Target number of sources.
            max_tiles: Baseline number of random tiles to pool; may be exceeded up
                to ``_STRATIFY_TILE_POOL_FACTOR × max_tiles`` when large sources are
                scarce (see the top-up logic below).
            exclude_str: Source IDs (as strings) to exclude.
            rng: Random state for reproducibility.

        Returns:
            Merged DataFrame of sampled rows, flattened in log source size.

        Raises:
            ValueError: If a catalogue lacks a diameter column. Cutana requires
                one for extraction, so its absence is a broken invariant.
        """
        # Group rows by tile (a tile may, in principle, span catalogue files) so
        # tiles can be drawn at random.  Read catalogues in *random* order (rng
        # is seeded from the labelled set, so the scanned files — and therefore
        # the drawn tiles — rotate across retrain cycles) and stop as soon as we
        # have discovered enough tiles to fill the pool.  Reading a random
        # *subset* rather than every catalogue is what keeps this bounded on
        # DR1's thousands of catalogue files; a single Q1 catalogue usually
        # already holds far more than enough tiles.  A UI-visible progress bar
        # makes the otherwise-silent scan legible while it runs.
        tile_target = _STRATIFY_TILE_POOL_FACTOR * max_tiles
        min_per_bin = max(1, n // self._cfg.cutana_size_stratify_bins)
        tile_parts: dict[str, list[pd.DataFrame]] = {}
        file_order = rng.permutation(len(self._catalogue_files))
        scanned = 0
        # Context-managed so the early break (the common case) still closes the
        # bar and emits a final line at the fraction actually scanned, rather
        # than leaving the UI progress tap frozen mid-count.
        with tqdm(
            total=len(file_order),
            desc="Selecting tiles (size stratification)",
            unit="catalogue",
            file=tqdm_logging_file(),
            mininterval=1,
        ) as scan_bar:
            for file_index in file_order:
                if len(tile_parts) >= tile_target:
                    break
                path, _ = self._catalogue_files[file_index]
                df = self._load_catalogue(path)
                scanned += 1
                scan_bar.update(1)
                if exclude_str:
                    df = df[~df["SourceID"].astype(str).isin(exclude_str)]
                # _load_catalogue guarantees fits_file_paths, so only emptiness matters.
                if df.empty:
                    continue
                if _resolve_size_column(df) is None:
                    raise ValueError(
                        f"Catalogue {path} has no source-size column "
                        f"(need one of {list(_SIZE_COLUMNS)}); size stratification "
                        "cannot run. Disable cfg.cutana_stratify_source_size or add a "
                        "diameter column to the catalogue."
                    )
                for tile, tile_df in df.groupby("fits_file_paths"):
                    tile_parts.setdefault(tile, []).append(tile_df)

        if not tile_parts:
            return pd.DataFrame()

        # Narrow the discovered tiles to the pool actually streamed.  Draw at
        # *random* (never by population, which would bias the pool toward the
        # densest fields). Start with max_tiles; if that draw is short on large
        # sources — the top flattened bin can't fill its quota — pool a few more
        # random tiles until it can, capped at _STRATIFY_TILE_POOL_FACTOR ×
        # max_tiles to bound how many FITS tiles get streamed.
        all_keys = list(tile_parts)
        rng.shuffle(all_keys)
        tile_cap = min(len(all_keys), _STRATIFY_TILE_POOL_FACTOR * max_tiles)
        # A size column is guaranteed above (missing-column tiles already raised),
        # so _finite_sizes never returns None here. Accumulate the per-tile arrays
        # in a list and concatenate only when the supply check actually runs (after
        # the max_tiles baseline), rather than rebuilding the array every iteration.
        chosen_keys: list[str] = []
        size_parts: list[np.ndarray] = []
        for key in all_keys:
            chosen_keys.append(key)
            size_parts.extend(_finite_sizes(df) for df in tile_parts[key])
            if len(chosen_keys) >= tile_cap:
                break
            if len(chosen_keys) >= max_tiles and _flat_plateau_supplied(
                np.concatenate(size_parts),
                self._cfg.cutana_size_stratify_bins,
                self._cfg.cutana_size_stratify_max_px,
                min_per_bin,
            ):
                break
        topped_up = len(chosen_keys) > max_tiles
        chosen = [(t, pd.concat(tile_parts[t], ignore_index=True)) for t in chosen_keys]
        logger.info(
            "Size stratification: pooled {} random tiles{} from {} tiles in "
            "{}/{} catalogues scanned ({} candidate sources) for uniform-in-size "
            "sampling",
            len(chosen),
            " (topped up for large-source supply)" if topped_up else "",
            len(tile_parts),
            scanned,
            len(self._catalogue_files),
            sum(len(df) for _tile, df in chosen),
        )
        # Pool the chosen tiles and draw once, globally, rather than per tile.
        # A single per-bin quota over the pooled sources fills the rare large-size
        # bins from *whichever* tiles hold those sources — a per-tile draw would
        # cap each bin at one tile's meagre large-source supply and undershoot.
        pool = pd.concat([df for _tile, df in chosen], ignore_index=True)
        self._record_population_sizes([pool])

        size_col = _resolve_size_column(pool)
        sizes = pd.to_numeric(pool[size_col], errors="coerce").to_numpy(dtype=float)
        indices = _quota_sample_indices(
            sizes,
            min(n, len(pool)),
            self._cfg.cutana_size_stratify_bins,
            self._cfg.cutana_size_stratify_max_px,
            rng,
        )
        return pool.iloc[indices].reset_index(drop=True)

    def _record_population_sizes(self, tile_frames: list[pd.DataFrame]) -> None:
        """Cache the candidate-population diameters of the drawn tiles.

        The population is every source in the tiles the sampler selected — the
        baseline the UI overlays the sampled pool against.  Called by both
        samplers *before* :meth:`_record_unlabeled_sizes` fires the histogram
        hook, so the population is available when the hook reads it.

        Args:
            tile_frames: Full source frames of the tiles drawn from (unsampled).
        """
        # Extract the diameter column per tile and concatenate the resulting
        # arrays, rather than concatenating the full multi-column frames (up to
        # cutana_max_unlabeled_tiles × 10k-100k rows) just to read one column.
        per_tile = [sizes for df in tile_frames if (sizes := _finite_sizes(df)) is not None]
        self._last_unlabeled_population_sizes = np.concatenate(per_tile) if per_tile else None

    def get_last_unlabeled_population_sizes(self) -> np.ndarray | None:
        """Return candidate-population diameters of the last batch's tiles.

        Returns:
            Finite diameters of every source in the tiles the last unlabeled
            batch was drawn from, or ``None`` if no batch has been sampled yet or
            the catalogue has no size column.
        """
        return self._last_unlabeled_population_sizes

    def _record_unlabeled_sizes(self, merged_df: pd.DataFrame) -> None:
        """Cache the finite source diameters of a sampled unlabeled batch.

        Stores ``None`` when the catalogue carries no size column so the
        training subprocess simply skips the size histogram for that source.
        Also logs the stratification summary so the two unlabeled-batch call
        sites can't drift on whether/how they report it.

        Args:
            merged_df: The sampled catalogue rows about to be streamed.
        """
        size_col = _resolve_size_column(merged_df)
        if size_col is None:
            self._last_unlabeled_sizes = None
            self._last_unlabeled_size_col = None
        else:
            sizes = pd.to_numeric(merged_df[size_col], errors="coerce").to_numpy(dtype=float)
            self._last_unlabeled_sizes = sizes[np.isfinite(sizes)]
            self._last_unlabeled_size_col = size_col

        # Fire the hook now — before ``_stream_cutouts`` runs — so the training
        # subprocess can surface the size histogram while cutouts stream (minutes)
        # rather than only after the pool is fully materialised.
        if self.unlabeled_sizes_callback is not None:
            self.unlabeled_sizes_callback()
        self._log_unlabeled_size_summary(len(merged_df))

    def get_last_unlabeled_sizes(self) -> np.ndarray | None:
        """Return finite source diameters from the most recent unlabeled batch.

        Returns:
            Array of diameters, or ``None`` if no batch has been sampled yet or
            the catalogue has no size column.
        """
        return self._last_unlabeled_sizes

    def get_last_unlabeled_size_unit(self) -> str | None:
        """Return the unit of the recorded sizes (``"pixel"`` or ``"arcsec"``).

        Returns:
            ``"pixel"`` for ``diameter_pixel``, ``"arcsec"`` for
            ``diameter_arcsec``, or ``None`` if no sized batch has been sampled.
        """
        if self._last_unlabeled_size_col == "diameter_pixel":
            return "pixel"
        if self._last_unlabeled_size_col == "diameter_arcsec":
            return "arcsec"
        return None

    def _log_unlabeled_size_summary(self, n_sources: int) -> None:
        """Log whether size stratification was applied and the resulting spread.

        Makes the session log state plainly whether the unlabeled pool was
        flattened in size (and how it came out) so a retrain's effect is
        visible without inspecting the histogram.

        Args:
            n_sources: Number of sources in the sampled batch.
        """
        stratify = self._cfg.cutana_stratify_source_size
        mode = "ON (uniform-in-size)" if stratify else "OFF (random)"
        sizes = self._last_unlabeled_sizes
        if sizes is None or len(sizes) == 0:
            # Reachable when stratification is off and the catalogue has no size
            # column, or (stratify on) when the column is present but all-NaN —
            # the missing-column case with stratify on raises earlier.
            logger.info(
                "Unlabeled size stratification {} — sampled {} sources "
                "(no usable diameters to summarise)",
                mode,
                n_sources,
            )
            return
        unit = self.get_last_unlabeled_size_unit() or "pixel"
        logger.info(
            "Unlabeled size stratification {} — {} sources, diameter "
            "median {:.0f}, mean {:.0f}, range [{:.0f}, {:.0f}] {}",
            mode,
            n_sources,
            float(np.median(sizes)),
            float(np.mean(sizes)),
            float(np.min(sizes)),
            float(np.max(sizes)),
            unit,
        )

    def _stream_cutouts(self, sub_catalogue_df: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
        """Create cutouts for sources in *sub_catalogue_df* using Cutana.

        Cutana is configured with identity channel weights, so its output is
        per-band normalised data with ``n_extensions`` channels. We apply the
        user's channel_combination here — the same post-processing used when
        reading from the labeled cache.

        Args:
            sub_catalogue_df: Filtered catalogue with sources to cut out.

        Returns:
            List of (source_id, image_array) tuples in HWC uint8.

        Raises:
            RuntimeError: If the installed Cutana release does not expose
                the ``skip_catalogue_validation`` flag, or if Cutana
                workers returned empty results for every batch (typically
                from a fitsbolt per-channel parameter length mismatch or
                unreachable FITS tiles).
        """
        # Sort by FITS tile so Cutana's greedy batch packer groups sources
        # sharing the same tile together, producing fewer larger batches
        # (each batch spawns a subprocess, so fewer = much faster).
        if "fits_file_paths" in sub_catalogue_df.columns:
            sub_catalogue_df = sub_catalogue_df.sort_values("fits_file_paths", ignore_index=True)

        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, "cutana_sub_catalogue.parquet")
        sub_catalogue_df.to_parquet(tmp_path, index=False)

        try:
            cutana_cfg = build_cutana_orchestrator_config(
                tmp_path, self._cfg, prune_unused_bands=True
            )
            orchestrator = StreamingOrchestrator(cutana_cfg)
            # Cap batch size for progress visibility; Cutana may further
            # split internally by FITS tile.
            stream_bs = min(self._cfg.cutana_streaming_batch_size, len(sub_catalogue_df))
            stream_bs = max(100, stream_bs)
            orchestrator.init_streaming(
                batch_size=stream_bs,
                write_to_disk=False,
                min_workers=self._cfg.cutana_min_workers,
                max_workers=self._cfg.cutana_max_workers,
            )

            results: list[tuple[str, np.ndarray]] = []
            n_batches = orchestrator.get_batch_count()
            # ``tqdm_logging_file`` keeps this meter visible in the UI as it
            # streams (see its module docstring for the why).
            for _ in tqdm(
                range(n_batches),
                desc="Streaming cutouts",
                unit="batch",
                total=n_batches,
                file=tqdm_logging_file(),
                mininterval=1,
            ):
                batch = orchestrator.next_batch()
                metadata = batch["metadata"]
                if len(metadata) == 0:
                    continue
                # One batched channel-combination matmul over the whole
                # next_batch stack instead of a per-image Python loop — the
                # same hot-path win applied to prediction in issue #458.
                # The helper turns the cutout list into an array itself, and
                # returns a contiguous stack, so its rows can be stored directly.
                combined = apply_channel_combination_to_cutana_batch(batch["cutouts"], self._cfg)
                for i, meta in enumerate(metadata):
                    results.append((str(meta["source_id"]), combined[i]))

            # Cutana's per-batch return is `{"cutouts": [], "metadata": []}`
            # on worker failure (see ``cutout_process.stream_cutouts_via_shm_pool``'s
            # fallback on empty results).  Callers downstream rely on a non-empty
            # batch — FixMatch's ``SSL_Dataset`` raises a misleading "No unlabeled
            # data were provided" assertion otherwise.  Surface the real cause
            # here instead so the session log points at the streaming stage.
            if n_batches > 0 and not results:
                raise RuntimeError(
                    f"Cutana streaming completed {n_batches} batch(es) but produced "
                    f"zero cutouts from {len(sub_catalogue_df)} source(s). "
                    "Cutana workers returned empty results — check the cutana "
                    "worker logs above (look for fitsbolt warnings about "
                    "per-channel parameter length, missing FITS files, or "
                    "catalogue schema issues)."
                )

            return results
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def get_labeled_images(self, label_df: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
        """Load cutouts for labeled source IDs across all catalogue files.

        Not available in lazy mode -- use a ``LabeledDataCache`` instead.

        Returns:
            List of (source_id, image_array) tuples.

        Raises:
            RuntimeError: If called in lazy mode.
        """
        if self._lazy:
            raise RuntimeError(
                "get_labeled_images() not available in lazy mode. Use a LabeledDataCache instead."
            )

        labeled_ids = set(str(x) for x in label_df["id"])
        if not labeled_ids:
            return []

        # Collect matching rows from all catalogues into one DataFrame
        merged_parts: list[pd.DataFrame] = []
        for path, _ in self._catalogue_files:
            df = self._load_catalogue(path)
            matching = df[df["SourceID"].astype(str).isin(labeled_ids)]
            if not matching.empty:
                merged_parts.append(matching)
                logger.debug(f"  {path.name}: {len(matching)} labeled matches")

        if not merged_parts:
            logger.info("CutanaSource: no labeled sources found in any catalogue")
            return []

        merged_df = pd.concat(merged_parts, ignore_index=True)
        logger.info(
            f"CutanaSource: streaming {len(merged_df)} labeled cutouts "
            f"(from {len(merged_parts)} catalogue files)"
        )
        return self._stream_cutouts(merged_df)

    def get_unlabeled_batch(
        self,
        n: int,
        exclude: set[str] | None = None,
    ) -> list[tuple[str, np.ndarray]]:
        """Load up to *n* random unlabeled cutouts across all catalogue files.

        Samples tile-aware across catalogues, dropping *exclude* (the labelled
        source IDs) from every tile before sampling in both eager and lazy mode —
        ``_sample_by_tiles`` loads each catalogue in full, so filtering by
        ``SourceID`` is free and keeps labelled sources out of the unlabelled
        pool.  Unlike the image/Zarr sources, Cutana sampling is tile-aware
        rather than position-indexed, so it takes no ``exclude_positions``.

        The RNG is seeded from ``cfg.seed`` mixed with the labelled set, so each
        active-learning cycle (which adds labels) draws a *fresh* unlabelled pool
        instead of the identical one a fixed seed would reproduce — while staying
        deterministic for a given label set.

        Args:
            n: Maximum number of cutouts to return.
            exclude: Source IDs to skip, also the per-cycle freshness key.

        Returns:
            List of (source_id, image_array) tuples.
        """
        exclude = exclude or set()
        seed = self._unlabeled_seed(exclude)
        exclude_str = set(str(x) for x in exclude)
        if self._lazy:
            return self._get_unlabeled_batch_lazy(n, seed, exclude_str)

        max_tiles = self._cfg.cutana_max_unlabeled_tiles
        rng = np.random.RandomState(seed)

        merged_df = self._sample_by_tiles(
            n,
            max_tiles,
            exclude_str,
            rng,
            stratify_size=self._cfg.cutana_stratify_source_size,
        )
        if merged_df.empty:
            return []
        self._record_unlabeled_sizes(merged_df)

        n_tiles = (
            merged_df["fits_file_paths"].nunique()
            if "fits_file_paths" in merged_df.columns
            else "?"
        )
        logger.info(
            f"CutanaSource: streaming {len(merged_df)} unlabeled cutouts "
            f"(from {n_tiles} FITS tiles)"
        )
        return self._stream_cutouts(merged_df)

    def _unlabeled_seed(self, exclude: set[str]) -> int:
        """Return an RNG seed that varies with the labelled set.

        Reseeding from a fixed ``cfg.seed`` on every call made each retrain cycle
        draw the identical unlabelled pool — and with size stratification the
        tile choice is deterministic too, so the model kept seeing the very same
        cutouts.  Mixing a stable hash of the (growing) labelled set into the
        seed gives each cycle a fresh pool while staying reproducible for a given
        label set.  ``hashlib`` (not the salted built-in ``hash``) keeps it
        reproducible across processes.

        Args:
            exclude: The labelled source IDs excluded from the unlabelled pool.

        Returns:
            A seed in ``[0, 2**32)``.
        """
        base = int(self._cfg.seed)
        if not exclude:
            return base
        digest = hashlib.sha1("\x00".join(sorted(str(x) for x in exclude)).encode()).hexdigest()
        return (base + int(digest[:8], 16)) % (2**32)

    def _get_unlabeled_batch_lazy(
        self,
        n: int,
        seed: int,
        exclude_str: set[str],
    ) -> list[tuple[str, np.ndarray]]:
        """Sample unlabeled cutouts via tile-aware random selection (lazy mode).

        Loads catalogues one by one, picks random FITS tiles, and samples
        sources from those tiles until *n* sources are collected.

        Args:
            n: Maximum number of cutouts to return.
            seed: RNG seed (already mixed with the labelled set by the caller).
            exclude_str: Source IDs (as strings) to drop from each tile before
                sampling — the labelled set, kept out of the unlabelled pool.

        Returns:
            List of (source_id, image_array) tuples.
        """
        rng = np.random.RandomState(seed)
        max_tiles = self._cfg.cutana_max_unlabeled_tiles

        # ``_sample_by_tiles`` loads each catalogue in full, so it can drop the
        # labelled ids by ``SourceID`` for free — no reason lazy mode should let
        # them resurface in the unlabelled pool.
        merged_df = self._sample_by_tiles(
            n,
            max_tiles,
            exclude_str,
            rng,
            stratify_size=self._cfg.cutana_stratify_source_size,
        )
        if merged_df.empty:
            return []
        self._record_unlabeled_sizes(merged_df)

        n_tiles = (
            merged_df["fits_file_paths"].nunique()
            if "fits_file_paths" in merged_df.columns
            else "?"
        )
        logger.info(
            f"CutanaSource(lazy): streaming {len(merged_df)} unlabeled cutouts "
            f"(from {n_tiles} FITS tiles)"
        )
        return self._stream_cutouts(merged_df)

    def get_total_count(self) -> int:
        """Return total number of sources across all catalogue files.

        Returns:
            Source count.
        """
        return self._total

    def detect_num_channels(self) -> int | None:
        """Return configured output channel count.

        Cutana channel count depends on the normalisation configuration,
        not the raw data, so we return the configured value.

        Returns:
            Number of output channels from config.
        """
        return self._cfg.normalisation.n_output_channels


def build_cutana_orchestrator_config(
    catalogue_path: str,
    cfg: DotMap,
    *,
    sub_catalogue_df: pd.DataFrame | None = None,
    do_only_cutout_extraction: bool = False,
    prune_unused_bands: bool = False,
) -> DotMap:
    """Build a Cutana orchestrator config from AnomalyMatch config.

    The orchestrator is configured with **identity channel weights** — one
    entry per resolved extension, each a 1.0-weighted one-hot vector of
    length ``n_extensions``.  This turns Cutana's internal
    ``combine_channels`` step into a no-op, so the orchestrator output is
    per-band normalised data with ``n_extensions`` channels.  The user's
    ``channel_combination`` matrix is then applied post-orchestrator via
    :func:`apply_channel_combination_to_cutana_image`, which keeps
    combination logic in one place shared with the read-from-cache and
    prediction-subprocess paths.

    This is the single source of truth for orchestrator config building.
    ``CutanaSource``, ``LabeledDataCache``, and ``prediction_process_cutana``
    all delegate here.

    Args:
        catalogue_path: Path to the sub-catalogue file.  Used for filter-name
            resolution (reads the first row's ``fits_file_paths``) unless
            *sub_catalogue_df* is provided; always assigned to
            ``cutana_cfg.source_catalogue`` for orchestrator callers.
        cfg: AnomalyMatch configuration.
        sub_catalogue_df: Optional in-memory catalogue override.  When given,
            filter names are read from its first row instead of hitting disk —
            the path to parquet roundtrip that the direct-cutout caller would
            otherwise need.
        do_only_cutout_extraction: When ``True``, Cutana skips channel
            combination, normalisation, and the resize to ``target_resolution``,
            returning each cutout at its true native pixel size (the detail
            view's native path).
        prune_unused_bands: When ``True``, skip reading FITS extensions whose
            ``channel_combination`` column is entirely zero — they contribute
            nothing to any output channel, so reading them is wasted I/O.  Only
            safe for ephemeral streaming (prediction, unlabeled training): the
            labeled cache and preview paths must keep **all** bands so the user
            can retune the combination without re-extracting, so they leave this
            ``False``.  The downstream combine matmul drops the matching all-zero
            matrix columns, making the pruned result identical to the full one.

    Returns:
        Cutana configuration object.

    Raises:
        ValueError: If ``cfg.fitsbolt_cfg`` is None — without it the user's
            normalisation method would not reach Cutana and it would silently
            apply a linear (CONVERSION_ONLY) stretch.
        RuntimeError: If the installed Cutana release does not expose the
            ``skip_catalogue_validation`` flag.
    """
    # Broken-invariant guard runs first, before any catalogue/disk work in
    # _resolve_extension_names.  Without an external fitsbolt config, Cutana
    # falls back to its own default normalisation (CONVERSION_ONLY / linear),
    # silently discarding the user's ASINH/LOG/ZSCALE choice — the model would
    # then be fed linearly-stretched cutouts it was never trained on.  Every
    # production caller populates cfg.fitsbolt_cfg first (training via
    # get_fitsbolt_config, prediction via sync_normalisation_from_checkpoint,
    # the labeled cache via _build_raw_extraction_cfg), so a None here is a bug:
    # fail hard rather than ship a linear-normalised run.
    if cfg.fitsbolt_cfg is None:
        raise ValueError(
            "cfg.fitsbolt_cfg is None when building the Cutana orchestrator config; "
            "the normalisation settings would not reach Cutana and it would fall back "
            "to a linear (CONVERSION_ONLY) stretch. Populate cfg.fitsbolt_cfg from the "
            "model checkpoint (prediction) or from cfg.normalisation via "
            "get_fitsbolt_config (training) before streaming cutouts."
        )

    # A non-None fitsbolt_cfg can still carry the wrong method: the stretch
    # reaching Cutana comes entirely from external_fitsbolt_cfg (built below
    # from cfg.fitsbolt_cfg), while cfg.normalisation carries the intended
    # method. Both production paths keep them in lock-step — training rebuilds
    # fitsbolt_cfg from cfg.normalisation via get_fitsbolt_config, prediction
    # writes both from the checkpoint via sync_normalisation_from_checkpoint —
    # so a disagreement means a caller populated one without the other. Fail
    # hard rather than silently stretch cutouts with a method the model was
    # never trained on (the None guard above only catches the missing case).
    fitsbolt_method = NormalisationMethod(cfg.fitsbolt_cfg["normalisation_method"])
    if fitsbolt_method != cfg.normalisation.normalisation_method:
        raise ValueError(
            f"cfg.fitsbolt_cfg normalisation_method ({fitsbolt_method.name}) disagrees with "
            f"cfg.normalisation.normalisation_method ({cfg.normalisation.normalisation_method.name}); "
            "the cutouts would be stretched with a method the model was not trained on. "
            "Rebuild cfg.fitsbolt_cfg from cfg.normalisation via get_fitsbolt_config (training) "
            "or re-sync from the checkpoint via sync_normalisation_from_checkpoint (prediction)."
        )

    cutana_cfg = cutana.get_default_config()
    cutana_cfg.source_catalogue = catalogue_path
    cutana_cfg.target_resolution = cfg.normalisation.image_size[0]
    cutana_cfg.padding_factor = cfg.normalisation.cutout_padding_factor
    # Cutana's own data_type controls the final dtype conversion —
    # external_fitsbolt_cfg.output_dtype is used only for fitsbolt's
    # per-channel normalisation and is not propagated to the batch output.
    cutana_cfg.data_type = np.dtype(cfg.normalisation.output_dtype).name

    extension_names, has_integer_indices = _resolve_extension_names(
        cfg, catalogue_path=catalogue_path, sub_catalogue_df=sub_catalogue_df
    )
    n_extensions_full = len(extension_names)
    band_keep_mask: np.ndarray | None = None
    if prune_unused_bands:
        extension_names, band_keep_mask = _prune_unused_extensions(cfg, extension_names)
    n_extensions = len(extension_names)

    # Multi-file catalogues (one FITS per band) store each file under its own
    # PRIMARY HDU, so fits_extensions must be ["PRIMARY"] when filter names
    # came from file-name inspection.  When the user supplied filter name
    # strings directly, trust those.
    if has_integer_indices or cfg.normalisation.fits_extension is None:
        cutana_cfg.fits_extensions = ["PRIMARY"]
    else:
        cutana_cfg.fits_extensions = extension_names

    cutana_cfg.selected_extensions = [{"name": name, "ext": "PRIMARY"} for name in extension_names]

    # Identity channel weights: each extension gets its own output channel,
    # fully isolated.  Cutana's combine_channels becomes a no-op; all
    # per-band data passes through to apply_normalisation, then to the cache.
    cutana_cfg.channel_weights = {
        name: [1.0 if k == j else 0.0 for k in range(n_extensions)]
        for j, name in enumerate(extension_names)
    }

    cutana_cfg.apply_flux_conversion = cfg.normalisation.apply_flux_conversion

    # Build a Cutana-specific fitsbolt config that matches the
    # per-band pass-through shape: n_expected=n_output=n_extensions,
    # no channel_combination.  Per-channel normalisation params must
    # be resized to n_extensions so ASINH scales/clips align with
    # bands; they live under ``normalisation.asinh_{scale,clip}`` in
    # the fitsbolt config (the top-level ``norm_asinh_*`` names only
    # exist on AnomalyMatch's ``cfg.normalisation``).  When the user
    # picked n_output_channels=3 for 4-band Euclid data, the
    # 3-element list silently slipped past a mismatched-key lookup
    # here and the Cutana worker produced zero cutouts.
    fitsbolt = DotMap(cfg.fitsbolt_cfg.toDict(), _dynamic=False)
    fitsbolt.channel_combination = None
    fitsbolt.n_output_channels = n_extensions
    fitsbolt.n_expected_channels = n_extensions
    for attr in ("asinh_scale", "asinh_clip"):
        val = fitsbolt.normalisation.get(attr)
        if isinstance(val, (list, tuple)):
            # Resize at the *full* band count so each surviving band keeps the
            # exact param it would get in an unpruned run, then drop the pruned
            # bands' entries by the same mask.  Truncating to the pruned count
            # instead would shift later bands onto earlier params for a
            # non-prefix prune, normalising them differently than the full run.
            resized = _resize_param_list(val, n_extensions_full)
            if band_keep_mask is not None:
                resized = [p for p, keep in zip(resized, band_keep_mask) if keep]
            fitsbolt.normalisation[attr] = resized
    # Cutana's apply_normalisation does ``if "crop_for_maximum_value"
    # in external_cfg.normalisation: crop_value = ...; crop_value[0]``
    # — an ``in`` check rather than a None check, so a key set to
    # ``None`` (the fitsbolt default when the user hasn't enabled
    # cropping) makes Cutana attempt ``None[0]`` and crash the whole
    # batch.  Drop the key explicitly when its value is None.
    if fitsbolt.normalisation.get("crop_for_maximum_value") is None:
        fitsbolt.normalisation.pop("crop_for_maximum_value", None)
    cutana_cfg.external_fitsbolt_cfg = fitsbolt

    # Catalogues reaching this point have already been validated by
    # AnomalyMatch (``source_validation._validate_cutana`` reads them
    # column-by-column, and ``_is_cutana_catalogue`` gate-keeps both
    # source selection and preview loading).  Cutana's per-row
    # re-validation is pure overhead on huge catalogues (29 M-row Q1
    # search ≈ minute), so we opt out.  The flag must be present in the
    # cutana release that ships with the project — fail loudly on an
    # older install instead of silently setting an ignored attribute.
    if "skip_catalogue_validation" not in cutana_cfg:
        raise RuntimeError(
            "Installed cutana release does not expose `skip_catalogue_validation`; "
            "upgrade cutana to a release that supports it."
        )
    cutana_cfg.skip_catalogue_validation = True

    # Native (unresized) extraction for the detail view: Cutana skips channel
    # combination, normalisation, and the resize to target_resolution, returning
    # the cutout at its true native pixel size.  The detail screen applies
    # normalisation + channel combination in-process afterwards, so the user
    # inspects real pixels rather than a fixed-resolution resample.
    if do_only_cutout_extraction:
        if "do_only_cutout_extraction" not in cutana_cfg:
            raise RuntimeError(
                "Installed cutana release does not expose `do_only_cutout_extraction`; "
                "upgrade cutana to a release that supports native cutout extraction."
            )
        cutana_cfg.do_only_cutout_extraction = True

    return cutana_cfg


DIRECT_CUTOUT_MAX_SOURCES = 5000
"""Threshold below which :func:`extract_cutouts_direct` is preferred over
``StreamingOrchestrator`` for catalogue-based cutout extraction.  Values above
this hit diminishing returns — the orchestrator's worker-pool parallelism
starts outweighing the subprocess/IPC setup cost.
"""


def _iter_batch_items(batch: dict):
    """Yield ``(source_id, image_array)`` pairs from a Cutana batch dict.

    ``create_cutouts_direct`` emits a stacked 4D ``(N, H, W, C)`` ndarray,
    and ``StreamingOrchestrator.next_batch`` emits a list of per-source
    3D ndarrays — both support positional indexing, so this helper
    presents them as a uniform stream.

    Args:
        batch: Dict with ``"cutouts"`` and ``"metadata"`` keys.

    Yields:
        ``(source_id_str, image_array)`` for each source in ``metadata``.
    """
    cutouts = batch["cutouts"]
    for i, meta in enumerate(batch["metadata"]):
        yield str(meta["source_id"]), cutouts[i]


# Whether the installed Cutana's ``create_cutouts_direct`` accepts the
# ``log_set_progress`` keyword.  Resolved once at import so the per-call wrapper
# stays a plain branch (see :func:`extract_cutouts_direct`).
_CREATE_CUTOUTS_SUPPORTS_LOG_FLAG = (
    "log_set_progress" in inspect.signature(cutana.create_cutouts_direct).parameters
)


def extract_cutouts_direct(
    sub_catalogue_df: pd.DataFrame, cfg: DotMap, *, do_only_cutout_extraction: bool = False
) -> list[dict]:
    """Run Cutana's in-process ``create_cutouts_direct`` on a filtered catalogue.

    Intended for small batches (< :data:`DIRECT_CUTOUT_MAX_SOURCES` rows).
    Avoids the subprocess spawn + IPC overhead of
    ``StreamingOrchestrator.next_batch`` so label-cache builds and UI previews
    return in seconds instead of tens of seconds.

    Passes *sub_catalogue_df* directly into
    :func:`build_cutana_orchestrator_config` so filter-name resolution stays
    in-memory; ``create_cutouts_direct`` itself ignores
    ``cutana_cfg.source_catalogue``.

    Args:
        sub_catalogue_df: Filtered catalogue with Cutana columns.
        cfg: AnomalyMatch configuration.
        do_only_cutout_extraction: When ``True``, Cutana returns each cutout at
            its native pixel size with no channel combination / normalisation /
            resize — the detail view's true-native path. ``batch["cutouts"]`` is
            then a list of per-source native arrays (``_iter_batch_items`` handles
            both that and the usual stacked ndarray).

    Returns:
        List of Cutana batch result dicts (one per FITS file set), each with
        ``"cutouts"``, ``"metadata"``, ``"wcs"``, ``"channel_names"`` keys.
        The list matches the contract of iterating
        ``StreamingOrchestrator.next_batch``.
    """
    cutana_cfg = build_cutana_orchestrator_config(
        catalogue_path="",
        cfg=cfg,
        sub_catalogue_df=sub_catalogue_df,
        do_only_cutout_extraction=do_only_cutout_extraction,
    )
    # This wrapper is invoked once per catalogue (label-cache build and UI
    # preview both loop over catalogues), so Cutana's per-FITS-set heartbeat —
    # always a small, repeated count — is pure noise here; opt out when the
    # installed Cutana exposes the flag.  The guard keeps us compatible with
    # older Cutana builds (CI installs from Cutana's develop branch, which may
    # lag this flag); progress is conveyed by our own per-catalogue logging.
    if _CREATE_CUTOUTS_SUPPORTS_LOG_FLAG:
        return cutana.create_cutouts_direct(sub_catalogue_df, cutana_cfg, log_set_progress=False)
    return cutana.create_cutouts_direct(sub_catalogue_df, cutana_cfg)


def _prune_unused_extensions(
    cfg: DotMap, extension_names: list[str]
) -> tuple[list[str], np.ndarray | None]:
    """Drop extensions whose ``channel_combination`` column is entirely zero.

    A band that every output channel weights at zero contributes nothing to the
    final cutout, so reading it is wasted I/O (each band is a separate windowed
    read over the FITS tile).  The order of the kept names is preserved so it
    stays aligned, column-for-column, with the matrix the downstream combine
    matmul applies (which drops the same all-zero columns).

    Returns *extension_names* unchanged (and a ``None`` mask) when no usable
    combination matrix is set or every band is used — pruning only ever removes
    bands the matrix already ignores, so it can never drop data the output depends
    on.  A matrix whose width doesn't match the band count is a broken invariant
    and raises rather than silently no-ops.

    Args:
        cfg: AnomalyMatch configuration holding ``normalisation.channel_combination``.
        extension_names: Ordered extension/filter names resolved for this run.

    Returns:
        Tuple ``(kept_names, keep_mask)``.  ``keep_mask`` is a boolean array over
        the original extension order (``True`` where the band survives), or
        ``None`` when nothing was pruned.  The caller uses it to subset per-band
        parameters by the same mask so the pruned run stays bit-identical.

    Raises:
        ValueError: If ``channel_combination`` is set but its column count differs
            from the number of resolved extensions — the matrix then cannot be
            paired with the bands and any combine would mix the wrong ones.
    """
    channel_combination = _get_channel_combination_array(cfg)
    if channel_combination is None:
        # No usable combination matrix (e.g. grayscale/single-channel): nothing
        # to prune, and no misconfiguration to report.
        return extension_names, None
    if channel_combination.shape[1] != len(extension_names):
        # The matrix maps input bands to output channels, so its width must equal
        # the resolved extension count. A mismatch means channel_combination does
        # not describe these bands — the columns can't be paired with extensions,
        # so any combine would mix the wrong bands (the downstream matmul only
        # catches this when the all-zero column drop happens *not* to realign).
        # This is a broken config invariant, so fail hard rather than guess.
        raise ValueError(
            f"channel_combination has {channel_combination.shape[1]} column(s) but "
            f"{len(extension_names)} FITS extension(s) were resolved for this run "
            f"({extension_names}); the matrix width must match the band count. "
            "Fix normalisation.channel_combination to match the catalogue's bands."
        )
    used = np.any(channel_combination != 0, axis=0)
    if not used.any():
        # Degenerate matrix: every band is weighted zero everywhere, so the
        # combined cutout is all-zero regardless of which bands we read. Pruning
        # to nothing would leave no extensions to extract, so keep them all and
        # let the (all-zero) combine run — but warn, because this almost
        # certainly means a misconfigured channel_combination rather than intent.
        logger.warning(
            "channel_combination weights every FITS band at zero — the combined "
            "cutout will be all-zero. Not pruning any extension; check "
            "normalisation.channel_combination."
        )
        return extension_names, None
    if used.all():
        # Every band feeds at least one output channel — nothing to prune.
        return extension_names, None
    kept = [name for name, keep in zip(extension_names, used) if keep]
    dropped = [name for name, keep in zip(extension_names, used) if not keep]
    # Debug rather than info: the selection is identical for every sub-catalogue
    # the training stream builds a config for, and the resolved
    # ``selected_extensions`` already appears in the orchestrator's debug line.
    logger.debug(
        "Skipping {} FITS extension(s) with zero channel_combination weight: {}",
        len(dropped),
        dropped,
    )
    return kept, used


def _resolve_extension_names(
    cfg: DotMap,
    *,
    catalogue_path: str | None = None,
    sub_catalogue_df: pd.DataFrame | None = None,
) -> tuple[list[str], bool]:
    """Resolve the ordered list of extension/filter names to load from Cutana.

    When ``fits_extension`` is ``None`` or contains integer indices, filter
    names are read from the catalogue's ``fits_file_paths`` column.  When
    it contains strings, they are used as-is.

    Args:
        cfg: AnomalyMatch configuration.
        catalogue_path: Path to a catalogue file or directory of catalogues.
            Used to resolve filter names from disk.  Ignored when
            *sub_catalogue_df* is provided.
        sub_catalogue_df: Optional in-memory catalogue override.  When given,
            filter names are derived from its first row's ``fits_file_paths``
            entry instead of reading from disk.

    Returns:
        Tuple of ``(extension_names, had_integer_or_none_fits_extension)``.
        ``extension_names`` is always non-empty.
    """
    fits_ext = cfg.normalisation.fits_extension
    if fits_ext is None:
        fits_ext_list: list = ["PRIMARY"]
    elif isinstance(fits_ext, (str, int)):
        fits_ext_list = [fits_ext]
    else:
        fits_ext_list = list(fits_ext)

    has_integer_indices = any(isinstance(e, int) for e in fits_ext_list)
    needs_resolution = has_integer_indices or fits_ext_list == ["PRIMARY"]

    if not needs_resolution:
        return [str(e) for e in fits_ext_list], False

    try:
        if sub_catalogue_df is not None:
            all_filter_names = _filter_names_from_fits_paths_entry(
                sub_catalogue_df["fits_file_paths"].iloc[0]
            )
        else:
            all_filter_names = resolve_catalogue_filter_names(catalogue_path)
    except Exception as exc:
        logger.warning("Could not resolve filter names from catalogue: {}", exc)
        all_filter_names = []

    if not all_filter_names:
        return [str(e) for e in fits_ext_list], has_integer_indices

    if has_integer_indices:
        integer_indices = [e for e in fits_ext_list if isinstance(e, int)]
        names = [all_filter_names[i] for i in integer_indices if i < len(all_filter_names)]
        if not names:
            return [str(e) for e in fits_ext_list], True
        return names, True

    return all_filter_names, False


def _filter_names_from_fits_paths_entry(fits_paths_entry) -> list[str]:
    """Parse filter names from a single ``fits_file_paths`` catalogue cell.

    Used by both the on-disk and in-memory filter-name resolution paths so
    their naming behaviour (including the ``Band_i`` fallback) stays identical.

    Args:
        fits_paths_entry: Raw value from the catalogue's ``fits_file_paths``
            column (typically a string of space/comma-separated paths).

    Returns:
        Ordered list of filter name strings.
    """
    fits_paths = parse_fits_file_paths(fits_paths_entry)
    names = [extract_filter_name(p) for p in fits_paths]
    if any(n == "UNKNOWN" for n in names):
        return [f"Band_{i}" for i in range(len(fits_paths))]
    return names


def _is_cutana_catalogue(path: Path) -> bool:
    """Check if a CSV/parquet file has required Cutana catalogue columns.

    Args:
        path: Path to a CSV or parquet file.

    Returns:
        True if the file contains ``SourceID`` and ``fits_file_paths`` columns.
    """
    required = {"SourceID", "fits_file_paths"}
    try:
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq  # noqa: PLC0415

            schema = pq.read_schema(path)
            return required <= set(schema.names)
        # CSV: read just the header
        with open(path) as f:
            header = f.readline().strip().split(",")
        return required <= set(header)
    except Exception as exc:
        logger.debug("Failed to check catalogue columns in {}: {}", path, exc)
        return False


def _count_catalogue_rows(path: Path) -> int:
    """Count rows in a catalogue file without loading it fully.

    Uses ``pyarrow.parquet.read_metadata`` for parquet files and line
    counting for CSV files.

    Args:
        path: Path to a CSV or parquet catalogue file.

    Returns:
        Number of data rows (excluding header for CSV).
    """
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq  # noqa: PLC0415

        return pq.read_metadata(path).num_rows
    # CSV: count lines minus header
    with open(path) as f:
        return sum(1 for _ in f) - 1


def resolve_catalogue_filter_names(catalogue_path: str) -> list[str]:
    """Resolve filter/band names from a single Cutana catalogue file or folder.

    Reads ``fits_file_paths`` from the first source and extracts filter
    names using Cutana's naming conventions.  When names are unrecognised,
    falls back to positional ``Band_0``, ``Band_1``, ... labels.

    Args:
        catalogue_path: Path to a catalogue file or directory containing
            catalogues.

    Returns:
        List of filter name strings, e.g. ``["VIS", "NIR-H", "NIR-J", "NIR-Y"]``.

    Raises:
        FileNotFoundError: If *catalogue_path* is a directory with no
            catalogue files.
    """
    if os.path.isfile(catalogue_path):
        first_file = catalogue_path
    else:
        entries = sorted(os.listdir(catalogue_path))
        candidates = [
            os.path.join(catalogue_path, fname)
            for fname in entries
            if fname.endswith((".parquet", ".csv"))
        ]
        if not candidates:
            raise FileNotFoundError(f"No catalogue files found in {catalogue_path}")
        first_file = candidates[0]

    if first_file.endswith(".parquet"):
        df = pd.read_parquet(first_file, columns=["fits_file_paths"]).head(1)
    else:
        df = pd.read_csv(first_file, usecols=["fits_file_paths"], nrows=1)

    return _filter_names_from_fits_paths_entry(df["fits_file_paths"].iloc[0])


def first_cutana_catalogue(folder: str) -> str | None:
    """Return the path of the first validated Cutana catalogue in *folder*.

    Reading the first ``.csv``/``.parquet`` blindly would trip over a stray
    non-catalogue file that sorts first (e.g. a ``labeled_data.csv`` label file
    living next to the catalogues).

    Args:
        folder: Directory to search.

    Returns:
        Path to the first file that passes ``_is_cutana_catalogue``, or ``None``
        when *folder* isn't a readable directory or holds no catalogue.
    """
    # Guard falsy paths explicitly: os.listdir(None) lists the cwd rather than
    # raising, which would then walk the wrong directory.
    if not folder:
        return None
    try:
        entries = sorted(os.listdir(folder))
    except OSError as exc:
        logger.warning("Cannot list {} to look for Cutana catalogues: {}", folder, exc)
        return None
    for fname in entries:
        fpath = Path(folder) / fname
        if fname.endswith((".parquet", ".csv")) and _is_cutana_catalogue(fpath):
            return str(fpath)
    return None


def detect_cutana_filter_names(folder: str, cfg: DotMap) -> list[str]:
    """Return the bands a Cutana source in *folder* will be extracted with.

    Resolves through :func:`_resolve_extension_names`, the resolver the
    orchestrator config and the labeled cache use, so the channel-combination
    editor is sized to exactly the bands extraction delivers.  A separate
    catalogue-only parse used to ignore an explicit ``fits_extension`` and
    returned nothing on a parse error, leaving the editor at its old width while
    extraction produced a different band count.

    Args:
        folder: Directory containing Cutana catalogue files.
        cfg: Configuration whose ``normalisation.fits_extension`` selects bands.

    Returns:
        Ordered band names, or an empty list when *folder* holds no Cutana
        catalogue.
    """
    catalogue_path = first_cutana_catalogue(folder)
    if catalogue_path is None:
        logger.warning("No Cutana catalogue found in {}; cannot resolve its bands", folder)
        return []
    extension_names, _ = _resolve_extension_names(cfg, catalogue_path=catalogue_path)
    return extension_names
