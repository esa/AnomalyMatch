#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

import warnings
from pathlib import Path

import pyarrow.parquet as pq
import pandas as pd
from loguru import logger
from cutana.catalogue_preprocessor import validate_catalogue_columns, check_fits_files_exist


def cutana_validate_files_and_count_sources(
    files: list[Path | str], chunk_size: int = 100_000
) -> tuple[list[Path], int, int]:
    """Validate catalogue files for cutana compatibility and count total number sources and total number of chunks to process.

    Args:
        files (list[Path | str]): list of file paths to validate (CSV or Parquet).
        chunk_size (int): number of rows to read per chunk.

    Returns:
        tuple[list[Path], int, int]: valid files, total number of sources, and total number of chunks.
    """

    def _validate_against_cutana(index: int, dataframe: pd.DataFrame) -> bool:
        # Check header once
        if index == 0:
            errors = validate_catalogue_columns(dataframe)
            if errors:
                return errors

        errors, _ = check_fits_files_exist(dataframe)
        if errors:
            return errors
        return []

    valid_files = []
    total_sources = 0
    total_chunks = 0

    for file in files:

        is_file_valid = True

        current_file_sources = 0
        current_file_chunks = 0

        if isinstance(file, Path):
            file_type = file.name.split(".")[-1]
        else:
            file_type = file.split(".")[-1]

        if file_type == "csv":
            for i, df in enumerate(pd.read_csv(file, chunksize=chunk_size)):

                errors = _validate_against_cutana(i, df)
                if errors:
                    current_file_sources = 0
                    current_file_chunks = 0
                    is_file_valid = False
                    msg = f"File {file} did not pass cutana compatibility check and will be skipped ({errors})"
                    logger.warning(msg)
                    warnings.warn(msg, RuntimeWarning)
                    break
                current_file_sources += len(df)
                current_file_chunks += 1

        elif file_type == "parquet":
            parquet_file = pq.ParquetFile(file)
            for i, batch in enumerate(parquet_file.iter_batches(batch_size=chunk_size)):
                df = batch.to_pandas()

                errors = _validate_against_cutana(i, df)
                if errors:
                    current_file_sources = 0
                    current_file_chunks = 0
                    is_file_valid = False
                    msg = f"File {file} did not pass cutana compatibility check and will be skipped ({errors})"
                    logger.warning(msg)
                    warnings.warn(msg, RuntimeWarning)
                    break
                current_file_sources += len(df)
                current_file_chunks += 1
        else:
            is_file_valid = False

        total_sources += current_file_sources
        total_chunks += current_file_chunks
        if is_file_valid:
            valid_files.append(file)

    return valid_files, total_sources, total_chunks


def cutana_buffer_generator(files: list[Path | str], buffer_path: Path, chunk_size: int = 100_000):
    """Generate temporary buffer files by reading catalogue files in chunks.

    Args:
        files (list[Path | str]): list of file paths to process (CSV or Parquet).
        buffer_path (Path): path where temporary buffer parquet will be written.
        chunk_size (int): number of rows to read per chunk.

    Yields:
        Path: path to the buffer file containing the current chunk.
    """
    buffer_path.parent.mkdir(parents=True, exist_ok=True)

    for file in files:

        if isinstance(file, Path):
            file_type = file.name.split(".")[-1]
        else:
            file_type = file.split(".")[-1]

        if file_type == "csv":
            for df in pd.read_csv(file, chunksize=chunk_size):
                df.to_parquet(buffer_path, index=False)
                yield buffer_path

        else:  # if not CSV then Parquet
            parquet_file = pq.ParquetFile(file)
            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                df = batch.to_pandas()
                df.to_parquet(buffer_path, index=False)
                yield buffer_path
