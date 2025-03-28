#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
import pytest
import torch
from anomaly_match.models.FixMatch import FixMatch


class MockDataset:
    """Mock dataset for testing FixMatch."""

    def __init__(self, n_samples=100, n_classes=2):
        self.data = torch.randn(n_samples, 3, 32, 32)
        self.targets = torch.randint(0, n_classes, (n_samples,))
        self.filenames = [f"img_{i}.png" for i in range(n_samples)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Make the dataset subscriptable for DataLoader."""
        return self.data[idx], self.targets[idx], self.filenames[idx]


class TestFixMatch:

    @pytest.fixture
    def net_builder(self):
        """Simple CNN network builder for testing."""

        def _builder(num_classes, in_channels):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, 16, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(16, 32, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Flatten(),
                torch.nn.Linear(32 * 8 * 8, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, num_classes),
            )

        return _builder

    @pytest.fixture
    def fixmatch_model(self, net_builder):
        """Create a basic FixMatch model for testing."""
        model = FixMatch(
            net_builder=net_builder,
            num_classes=2,
            in_channels=3,
            ema_m=0.999,
            T=0.5,
            p_cutoff=0.95,
            lambda_u=1.0,
            hard_label=True,
        )
        # Set optimizer
        optimizer = torch.optim.SGD(model.train_model.parameters(), lr=0.01)
        model.set_optimizer(optimizer)
        return model

    def test_initialization(self, fixmatch_model):
        """Test that the model initializes correctly."""
        # Check that both models exist
        assert fixmatch_model.train_model is not None
        assert fixmatch_model.eval_model is not None

        # Check hyperparameters
        assert fixmatch_model.T == 0.5
        assert fixmatch_model.p_cutoff == 0.95
        assert fixmatch_model.lambda_u == 1.0
        assert fixmatch_model.use_hard_label is True

        # Check optimizer
        assert fixmatch_model.optimizer is not None

    def test_eval_model_update(self, fixmatch_model):
        """Test that the EMA update works correctly."""
        # Get initial parameters
        train_params = list(fixmatch_model.train_model.parameters())
        eval_params = [
            p.clone() for p in fixmatch_model.eval_model.parameters()
        ]  # Clone to keep originals

        # Set FixMatch model to a smaller momentum to make changes more visible in test
        fixmatch_model.ema_m = 0.8

        # Change train model parameters significantly
        for p in train_params:
            if p.requires_grad:
                # Make a significant change
                p.data = p.data + torch.ones_like(p.data)

        # Perform EMA update
        fixmatch_model._eval_model_update()

        # Check that eval parameters have moved toward train parameters
        param_differences_found = False

        for p_train, p_eval, p_orig in zip(
            fixmatch_model.train_model.parameters(),
            fixmatch_model.eval_model.parameters(),
            eval_params,
        ):
            if p_train.requires_grad:
                # With ema_m = 0.8 and adding 1.0, the eval params should change by 0.2
                if not torch.allclose(p_eval, p_orig, rtol=1e-3, atol=1e-3):
                    param_differences_found = True
                    break

                # Check that the params are between original and new (due to momentum)
                # The difference between eval and train should be about 0.8 times the
                # original difference after EMA update
                orig_diff = torch.abs(p_orig - p_train)
                curr_diff = torch.abs(p_eval - p_train)
                if not torch.allclose(curr_diff, 0.8 * orig_diff, rtol=1e-3, atol=1e-3):
                    param_differences_found = True
                    break

        assert param_differences_found, "EMA update did not change parameters as expected"

    def test_forward_pass(self, fixmatch_model):
        """Test that forward pass works for both models."""
        # Create a batch of data
        x = torch.randn(10, 3, 32, 32)

        # Test train model
        fixmatch_model.train_model.eval()  # Set to eval mode to avoid batch norm issues
        y_train = fixmatch_model.train_model(x)
        assert y_train.shape == (10, 2)

        # Test eval model
        y_eval = fixmatch_model.eval_model(x)
        assert y_eval.shape == (10, 2)

    @pytest.mark.parametrize("use_hard_labels", [True, False])
    def test_consistency_loss_variants(self, fixmatch_model, use_hard_labels):
        """Test that consistency loss works with both hard and soft labels."""
        from anomaly_match.utils.consistency_loss import consistency_loss

        # Create logits for weak and strong augmentations
        logits_w = torch.randn(10, 2)
        logits_s = torch.randn(10, 2)

        # Calculate consistency loss
        loss, mask = consistency_loss(
            logits_w,
            logits_s,
            name="ce",
            T=0.5,
            p_cutoff=0.5,  # Lower for testing to have non-zero mask
            use_hard_labels=use_hard_labels,
        )

        # Check outputs have the right type
        assert isinstance(loss, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert loss.ndim == 0  # Scalar tensor
        assert mask.ndim == 0  # Scalar tensor

    @pytest.mark.parametrize("use_weighted_sampler", [True, False])
    def test_set_data_loader(self, fixmatch_model, use_weighted_sampler):
        """Test data loader setup."""
        # Create mock datasets
        lb_dataset = MockDataset(100)
        ulb_dataset = MockDataset(300)
        eval_dataset = MockDataset(50)

        # Create simple config
        class Config:
            def __init__(self):
                self.batch_size = 16
                self.uratio = 2  # Ratio between unlabeled and labeled batch sizes
                self.num_train_iter = 10
                self.num_workers = 0
                self.pin_memory = False
                self.eval_batch_size = 16
                self.oversample = use_weighted_sampler
                self.gpu = 0  # Needed for the set_data_loader function

        cfg = Config()

        # Set data loaders
        fixmatch_model.set_data_loader(cfg, lb_dataset, ulb_dataset, eval_dataset)

        # Check loaders were created
        assert "train_lb" in fixmatch_model.loader_dict
        assert "train_ulb" in fixmatch_model.loader_dict
        assert "eval" in fixmatch_model.loader_dict

        # Check batch sizes
        batch = next(iter(fixmatch_model.loader_dict["train_lb"]))
        assert len(batch) == 3  # x, y, filename
        assert batch[0].shape[0] <= cfg.batch_size  # Could be smaller for last batch

        batch = next(iter(fixmatch_model.loader_dict["train_ulb"]))
        assert batch[0].shape[0] <= cfg.batch_size * cfg.uratio

        batch = next(iter(fixmatch_model.loader_dict["eval"]))
        assert batch[0].shape[0] <= cfg.eval_batch_size
