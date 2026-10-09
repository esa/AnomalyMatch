#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for the shared prefetch helper.

``pipelined_batches`` overlaps upcoming batch loads with the current batch's
consumption (``lookahead`` batches in flight, default 1; Cutana passes 2) and is shared by all
three prediction processes, so its ordering, skip, shutdown, thread-affinity
and error-propagation contracts are worth pinning down on the CPU (the GPU
subprocesses can't run in CI).
"""

import threading

import pytest

from prediction_utils import pipelined_batches


def test_yields_index_and_payload_in_order():
    out = list(pipelined_batches(4, prepare=lambda i: i, load=lambda plan: plan * 10))
    assert out == [(0, 0), (1, 10), (2, 20), (3, 30)]


def test_none_plan_skips_batch():
    # Odd indices are "already scored" — prepare returns None, so they're skipped
    # but the surviving indices keep their original positions.
    out = list(
        pipelined_batches(
            5,
            prepare=lambda i: None if i % 2 else i,
            load=lambda plan: plan,
        )
    )
    assert out == [(0, 0), (2, 2), (4, 4)]


def test_load_runs_on_single_background_worker():
    caller = threading.current_thread().name
    prepare_threads = []
    load_threads = []

    def prepare(index):
        prepare_threads.append(threading.current_thread().name)
        return index

    def load(plan):
        load_threads.append(threading.current_thread().name)
        return plan

    list(pipelined_batches(3, prepare=prepare, load=load))

    # prepare stays on the caller's thread (where the DB connection lives)
    assert set(prepare_threads) == {caller}
    # load runs on exactly one background worker, never the caller
    assert caller not in load_threads
    assert len(set(load_threads)) == 1


def test_load_exception_propagates_to_consumer():
    def load(plan):
        if plan == 2:
            raise ValueError("boom on batch 2")
        return plan

    seen = []
    with pytest.raises(ValueError, match="boom on batch 2"):
        for _index, payload in pipelined_batches(5, prepare=lambda i: i, load=load):
            seen.append(payload)

    # Batches before the failure were delivered; the failing one aborts iteration.
    assert seen == [0, 1]


def test_should_stop_halts_preparation():
    prepared = []

    def prepare(index):
        prepared.append(index)
        return index

    # Stop once we've consumed two batches.
    consumed = []

    def should_stop():
        return len(consumed) >= 2

    for _index, payload in pipelined_batches(
        100, prepare=prepare, load=lambda plan: plan, should_stop=should_stop, lookahead=2
    ):
        consumed.append(payload)

    # Depth-2 prefetch: when should_stop flips after batch 1 is consumed, batches
    # 2 and 3 were already in flight and still complete — but nothing past them is
    # prepared.
    assert consumed == [0, 1, 2, 3]
    assert max(prepared) <= 3  # stopped early, did not walk all 100 indices


def test_lookahead_one_prepares_a_single_batch_ahead():
    prepared = []
    consumed = []

    def prepare(index):
        prepared.append(index)
        return index

    def should_stop():
        return len(consumed) >= 2

    for _index, payload in pipelined_batches(
        100, prepare=prepare, load=lambda plan: plan, should_stop=should_stop, lookahead=1
    ):
        consumed.append(payload)

    # lookahead=1 keeps only one batch in flight, so a single extra batch
    # completes after the stop flips.
    assert consumed == [0, 1, 2]
    assert max(prepared) <= 2


def test_lookahead_must_be_positive():
    with pytest.raises(ValueError, match="lookahead must be >= 1"):
        list(pipelined_batches(3, prepare=lambda i: i, load=lambda p: p, lookahead=0))
