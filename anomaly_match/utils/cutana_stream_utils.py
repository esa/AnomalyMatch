#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Cutana streaming data utilities."""

from __future__ import annotations

import math
import warnings
from collections.abc import Generator
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from cutana.catalogue_preprocessor import validate_catalogue_columns
from loguru import logger


def cutana_validate_files_and_count_sources(
    files: list[Path | str], chunk_size: int = 100_000
) -> tuple[list[Path], int, int]:
    """Validate catalogue files for cutana compatibility and count sources.

    Only checks column schema (first chunk of first file).  Row counts are
    read from parquet metadata when possible to avoid scanning every row.

    Args:
        files: list of file paths to validate (CSV or Parquet).
        chunk_size: number of rows to read per chunk.

    Returns:
        valid files, total number of sources, and total number of chunks.
    """
    # Schema is validated once from the first valid file (all catalogue files
    # are expected to share the same column layout).
    columns_validated = False
    valid_files = []
    total_sources = 0
    total_chunks = 0

    for file in files:
        file_str = str(file)
        file_type = file_str.rsplit(".", 1)[-1].lower()

        if file_type == "parquet":
            try:
                parquet_file = pq.ParquetFile(file_str)

                # Validate columns once from the first file's schema
                if not columns_validated:
                    first_batch = next(parquet_file.iter_batches(batch_size=1))
                    errors = validate_catalogue_columns(first_batch.to_pandas())
                    if errors:
                        msg = f"File {file} did not pass cutana column check ({errors})"
                        logger.warning(msg)
                        warnings.warn(msg, RuntimeWarning)
                        continue
                    columns_validated = True

                num_rows = parquet_file.metadata.num_rows
                total_sources += num_rows
                total_chunks += math.ceil(num_rows / chunk_size)
                valid_files.append(file)
            except Exception as e:
                logger.warning(f"Could not read parquet file {file}: {e}")

        elif file_type == "csv":
            current_sources = 0
            current_chunks = 0
            is_valid = True
            for i, df in enumerate(pd.read_csv(file_str, chunksize=chunk_size)):
                if not columns_validated:
                    errors = validate_catalogue_columns(df)
                    if errors:
                        msg = f"File {file} did not pass cutana column check ({errors})"
                        logger.warning(msg)
                        warnings.warn(msg, RuntimeWarning)
                        is_valid = False
                        break
                    columns_validated = True
                current_sources += len(df)
                current_chunks += 1
            if is_valid:
                total_sources += current_sources
                total_chunks += current_chunks
                valid_files.append(file)
        else:
            logger.warning(f"Unsupported file type '{file_type}' for {file}, skipping")

    return valid_files, total_sources, total_chunks


def cutana_buffer_generator(
    files: list[Path | str], buffer_path: Path, chunk_size: int = 100_000
) -> Generator[Path, None, None]:
    """Generate temporary buffer files by reading catalogue files in chunks.

    Each chunk is written to a distinct ``<buffer_path>.N.parquet`` file
    so a just-exited prediction subprocess can't race the next chunk's
    write on the same inode (previously caused the chunk loop to hang
    after chunk 1 on NFS).  The index-suffixed files are cleaned up as
    later chunks are written so total disk usage stays at one chunk.

    Args:
        files: list of file paths to process (CSV or Parquet).
        buffer_path: base path; per-chunk files are written alongside
            with an index suffix.
        chunk_size: number of rows to read per chunk.

    Yields:
        Path: path to the buffer file containing the current chunk.
    """
    base = Path(buffer_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    previous: Path | None = None
    chunk_idx = 0

    def _write_chunk(df: pd.DataFrame) -> Path:
        nonlocal previous, chunk_idx
        out = base.with_name(f"{base.stem}.{chunk_idx}{base.suffix}")
        df.to_parquet(out, index=False)
        if previous is not None and previous != out and previous.exists():
            try:
                previous.unlink()
            except OSError as exc:
                logger.debug("Could not delete stale chunk buffer {}: {}", previous, exc)
        previous = out
        chunk_idx += 1
        return out

    for file in files:
        if isinstance(file, Path):
            file_type = file.name.split(".")[-1].lower()
        else:
            file_type = file.split(".")[-1].lower()

        if file_type == "csv":
            for df in pd.read_csv(file, chunksize=chunk_size):
                yield _write_chunk(df)

        else:  # if not CSV then Parquet
            parquet_file = pq.ParquetFile(file)
            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                yield _write_chunk(batch.to_pandas())

    # Clean up the last chunk after the consumer is done.
    if previous is not None and previous.exists():
        try:
            previous.unlink()
        except OSError as exc:
            logger.debug("Could not delete final chunk buffer {}: {}", previous, exc)
