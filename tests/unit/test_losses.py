#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for accuracy, cross_entropy_loss, and consistency_loss."""

import pytest
import torch

from anomaly_match.utils.accuracy import accuracy
from anomaly_match.utils.consistency_loss import consistency_loss
from anomaly_match.utils.cross_entropy_loss import cross_entropy_loss


class TestAccuracy:
    def test_perfect_predictions(self):
        output = torch.tensor([[10.0, -10.0], [-10.0, 10.0], [10.0, -10.0]])
        target = torch.tensor([0, 1, 0])
        [top1] = accuracy(output, target, topk=(1,))
        assert top1.item() == 100.0

    def test_zero_accuracy(self):
        output = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
        target = torch.tensor([1, 0])
        [top1] = accuracy(output, target, topk=(1,))
        assert top1.item() == 0.0

    def test_partial_accuracy(self):
        output = torch.tensor([[10.0, -10.0], [-10.0, 10.0], [10.0, -10.0], [-10.0, 10.0]])
        target = torch.tensor([0, 1, 1, 0])
        [top1] = accuracy(output, target, topk=(1,))
        assert top1.item() == 50.0

    def test_batch_size_one(self):
        output = torch.tensor([[5.0, 1.0]])
        target = torch.tensor([0])
        [top1] = accuracy(output, target, topk=(1,))
        assert top1.item() == 100.0


class TestCrossEntropyLoss:
    def test_hard_labels_shape(self):
        logits = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 0, 1])
        loss = cross_entropy_loss(logits, targets, use_hard_labels=True, reduction="none")
        assert loss.shape == (4,)

    def test_hard_labels_mean_reduction(self):
        logits = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 0, 1])
        loss = cross_entropy_loss(logits, targets, use_hard_labels=True, reduction="mean")
        assert loss.ndim == 0  # scalar

    def test_hard_labels_non_negative(self):
        logits = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 0, 1])
        loss = cross_entropy_loss(logits, targets, use_hard_labels=True, reduction="none")
        assert (loss >= 0).all()

    def test_soft_labels_shape(self):
        logits = torch.randn(4, 2)
        targets = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.7, 0.3], [0.3, 0.7]])
        loss = cross_entropy_loss(logits, targets, use_hard_labels=False)
        assert loss.shape == (4,)

    def test_soft_labels_non_negative(self):
        logits = torch.randn(4, 2)
        targets = torch.softmax(torch.randn(4, 2), dim=-1)
        loss = cross_entropy_loss(logits, targets, use_hard_labels=False)
        assert (loss >= 0).all()

    def test_soft_labels_shape_mismatch_raises(self):
        logits = torch.randn(4, 2)
        targets = torch.randn(4, 3)
        with pytest.raises(AssertionError):
            cross_entropy_loss(logits, targets, use_hard_labels=False)


class TestConsistencyLoss:
    def test_l2_loss(self):
        logits_w = torch.randn(4, 2)
        logits_s = torch.randn(4, 2)
        loss = consistency_loss(logits_w, logits_s, name="L2")
        assert loss.ndim == 0  # scalar
        assert loss >= 0

    def test_ce_loss_returns_tuple(self):
        logits_w = torch.randn(4, 2)
        logits_s = torch.randn(4, 2)
        result = consistency_loss(logits_w, logits_s, name="ce", p_cutoff=0.0)
        assert isinstance(result, tuple)
        assert len(result) == 2
        masked_loss, mask_ratio = result
        assert masked_loss.ndim == 0
        assert mask_ratio.ndim == 0

    def test_ce_hard_labels(self):
        logits_w = torch.randn(4, 2)
        logits_s = torch.randn(4, 2)
        masked_loss, mask_ratio = consistency_loss(
            logits_w, logits_s, name="ce", use_hard_labels=True, p_cutoff=0.0
        )
        assert masked_loss >= 0
        assert mask_ratio == 1.0  # p_cutoff=0 means all samples pass

    def test_ce_soft_labels(self):
        logits_w = torch.randn(4, 2)
        logits_s = torch.randn(4, 2)
        masked_loss, mask_ratio = consistency_loss(
            logits_w, logits_s, name="ce", use_hard_labels=False, p_cutoff=0.0
        )
        assert masked_loss >= 0
        assert mask_ratio == 1.0

    def test_ce_high_cutoff_masks_all(self):
        # Uniform logits = 0.5 probability, cutoff at 0.99 should mask all
        logits_w = torch.zeros(4, 2)
        logits_s = torch.randn(4, 2)
        masked_loss, mask_ratio = consistency_loss(logits_w, logits_s, name="ce", p_cutoff=0.99)
        assert mask_ratio == 0.0

    def test_ce_detaches_weak_logits(self):
        logits_w = torch.randn(4, 2, requires_grad=True)
        logits_s = torch.randn(4, 2, requires_grad=True)
        masked_loss, _ = consistency_loss(logits_w, logits_s, name="ce", p_cutoff=0.0)
        masked_loss.backward()
        # Gradient should only flow through logits_s, not logits_w
        assert logits_w.grad is None
        assert logits_s.grad is not None

    def test_invalid_loss_name_raises(self):
        logits_w = torch.randn(4, 2)
        logits_s = torch.randn(4, 2)
        with pytest.raises(AssertionError):
            consistency_loss(logits_w, logits_s, name="invalid")
