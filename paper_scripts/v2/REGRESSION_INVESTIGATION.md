[//]: # (Copyright &#40;c&#41; European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file 'LICENCE.txt'.)

# v2 vs v1 benchmark — regression investigation (handoff notes)

## ✅ SOLVED (2026-07-06): classifier-head init + bn_momentum, both from the timm swap
Root cause of the v2 regression (AUROC 0.81 vs v1 0.96), confirmed by three
independent audits + bit-exact diagnostics:

1. **Classifier-head initialisation (the mean-gap cause).** timm inits the head with
   TF-EfficientNet `_init_weight_goog` (`1/sqrt(fan_in+fan_out)`, tuned for 1000
   classes) → for AnomalyMatch's 2-class head the weight std is ~0.4, **~25x** the
   PyTorch-default `1/sqrt(fan_in)` (~0.016) the old `efficientnet_lite_pytorch`
   head used. The over-scaled head is overconfident at init, so **~100% of unlabeled
   images clear FixMatch's `p_cutoff`=0.95 with random pseudo-labels from step 1**,
   burning a garbage boundary into the backbone. (v1 head: 0% clear the cutoff →
   pseudo-labels ramp in correctly.) Fix: `model.get_classifier().reset_parameters()`
   after `timm.create_model` (`get_net_builder.py`).
2. **`cfg.bn_momentum` never applied (variance + partial mean).** Computed
   (`=1-ema_m=0.01`) and validated but never set on any BN layer; timm defaults to
   0.1 → BN stats track 10x too fast at batch 16. Fix: apply it to both models'
   `_BatchNorm` layers (`training_process.py`).
3. **`cfg.seed` never applied (reproducibility/variance).** No `torch.manual_seed`
   in the training path. Fix: seed torch/numpy at training-subprocess start.

Backbone weights are byte-identical (timm `tf_efficientnet_lite0.in1k` == the v1
`efficientnet_lite0_pytorch_model` port); eval-forward and all gradients are
bit-identical given the same head. So **no weight swap / no git-LFS asset / no
dependency change** — the fix stays on timm + safetensors. Ruled out: EMA/buffer
handling (correct), normalisation ([-1,1] a wash), drop_path/drop_rate (no effect),
Conv2dSame/BatchNormAct2d numerics (bit-identical). Validation of the combined fix
(hourglass+piano) in `results/validate_realfix/`.

---


Status snapshot for resuming after context compaction. Branch
`v2-benchmark-validation`, PR **#539** (base `fix/labeled-cache-label-map-and-append-hash`).

## What we're doing
Validating that AnomalyMatch **v2** (subprocess architecture) reproduces the
original **v1** paper numbers (arXiv:2505.03509). Harness lives in
`paper_scripts/v2/`. The v1 paper results are at
`/media/team_workspaces/AnomalyMatch/paper_results/FullCorrectedOutput`. v2
outputs + comparisons go to `/media/team_workspaces/AnomalyMatch/paper_results_v2`.

miniImageNet class → label_idx: **hourglass=57, piano=85, guitar=48, printer=68,
orange=95** (verified by rendering images; the CSV only has integer labels).

## Harness (committed)
- `run_benchmark.py` — static label-count sweep (train once per budget).
- `run_active_learning.py` — **the fair comparison**: reproduces v1's
  train→relabel→retrain loop (start 5/495; each cycle train N iters, score all,
  relabel top-10 true anomalies + top-10 false-positive nominals via
  `select_new_labels`, repeat). Cycle k ↔ v1 iteration k. Retrains fresh from the
  pretrained backbone each cycle (use `--iters-per-cycle 100` to match paper's
  per-cycle budget).
- `compare_v1_v2.py` — overlays v2 vs v1 detection-efficiency curves; auto-detects
  budget-sweep (`budget_idx`) vs AL (`cycle`) outputs. Writes to
  `paper_results_v2/comparisons/`.
- `paper_plots.py` + `plot_colors.py` — the v1 paper plotting code, reused
  verbatim (only fixed 2 imports + inlined `save_plot_data`). Excluded from ruff.

## Runs on disk (all under `paper_scripts/v2/results/`, gitignored)
- `miniimagenet_al/` — **active-learning run, IN PROGRESS** (300 iters/cycle,
  3 cycles, all 5 classes). Hourglass done; piano/guitar/printer/orange pending.
  PID in `miniimagenet_al/run.pid`. Log: `miniimagenet_al/active_learning.log`.
- `miniimagenet_iter200/` — static budget sweep, 200 iters (hourglass complete +
  partial). `miniimagenet_budgetsweep_iter300/` — static sweep, 300 iters (partial).

## KEY FINDING: confirmed regression (not fine)
Fair apples-to-apples (AL loop, identical per-cycle label counts, seed 42):

| stage | labels | v2 AUROC | v1 AUROC | v2 AUPRC | v1 AUPRC | v2 top-1% | v1 top-1% |
|---|---|---|---|---|---|---|---|
| cycle 1 | 5 anom / 495 nom | **0.814** | 0.961 | 0.217 | 0.807 | 23% | — |
| cycle 2 | 15 anom / 505 nom | 0.841 | — | 0.329 | — | 33% | — |
| cycle 3 (final) | 25 anom / 515 nom | **0.882** | 0.970 | 0.394 | 0.838 | 40% | 76% |

Cycle 1 uses the **identical** initial 5/495 labels as v1 `iteration_1`, so the
0.81 vs 0.96 gap is a genuine pipeline difference — not label count, not active
learning. AUPRC is less than half v1's throughout. This is a real regression.

(Static sweep for reference — 300 iters: hourglass 5 anom→0.80, 20→0.94, 35→0.94;
so more static anomalies help but never reach v1. AL with 25 selected anomalies
= 0.882, lower than static 35 random = 0.944, i.e. our AL selection isn't adding
the value v1's did — secondary issue.)

## Ruled OUT
1. **Hyperparameters** — v1's saved `config.toml`
   (`.../benchmark_miniimagenet_anomaly57_*/config.toml`) is byte-identical to v2
   defaults: lr 0.0075, batch 16, uratio 5, p_cutoff 0.95, ema_m 0.99, wd 7.5e-4,
   SGD/momentum 0.9, hard_label, temperature 0.5, ulb_loss_ratio 1.0,
   bn_momentum 0.01, efficientnet-lite0, pretrained, `CONVERSION_ONLY`
   normalisation, size 224, N_to_load 10000, oversample true, test_ratio 0.
2. **EMA scoring** — v2 checkpoint saves both `train_model` and `eval_model`;
   prediction loads `checkpoint["eval_model"]` (`prediction_utils.py:417`), i.e.
   the EMA model, same as v1 (`session.model.eval_model`).

## ROOT CAUSE IDENTIFIED (2026-07-06): missing input normalisation

The efficientnet-lite0 implementation was swapped v1→v2 (commit
`9f951f45 feat(deps): replace efficientnet packages with timm (#266)`):
- **v1**: `efficientnet_lite_pytorch.EfficientNet.from_pretrained` — params named
  `_blocks.0._bn1.*` (lukemelas fork).
- **v2**: `timm.create_model("tf_efficientnet_lite0.in1k")` — params named
  `blocks.0.0.bn1.*`. (This mismatch blocked the naive cross-scoring graft.)

timm's `tf_efficientnet_lite0.in1k` declares `pretrained_cfg` **mean=(0.5,0.5,0.5),
std=(0.5,0.5,0.5)** → it expects inputs in **[-1, 1]**. But v2's transforms
(`image_processing/transforms.py`, weak/strong/prediction) end in `ToTensor()` →
**[0, 1]** with **no `Normalize`**. So the pretrained timm backbone is fed inputs
shifted +0.5 and at half contrast → degraded transfer. This is consistent with v2
barely beating v1's *untrained* pretrained baseline (0.797) after 300 iters.
`git log -S Normalize` shows transforms.py never had a Normalize, so v1 also fed
[0,1] — but v1's lukemelas checkpoint tolerated [0,1], whereas the timm tf_ port
does not (it wants Inception-style [-1,1]).

**Result (cycle-1, n=1): REFUTED as a simple win.** `norm_fix_experiment.py`
retrained hourglass 5/495 (seed 42, 300 iters) with `Normalize([0.5]*3,[0.5]*3)`
(→ [-1,1]) on all three transforms. AUROC = **0.6648** — *worse* than the [0,1]
baseline (0.8142). So matching timm's declared [-1,1] range did NOT help on the
cold start; [0,1] is the better single-cycle choice.

BUT the cold start (5/495) is high-variance (cf. clean [0,1] piano cycle 1 = 0.861
vs hourglass cycle 1 = 0.814 — 0.05 from class alone), so a single cycle-1 draw is
not decisive. **In flight**: full 3-cycle AL with [-1,1] for hourglass+piano
(`results/miniimagenet_al_neg11/`) to compare across cycles vs the clean [0,1] AL
(`results/miniimagenet_al/`, hourglass 0.814/0.841/0.882, piano 0.861/0.872/0.833)
and v1 (0.961/–/0.970). If [-1,1] does not close the gap over cycles, normalisation
is not the regression and the model swap (timm tf_efficientnet_lite0 vs the v1
`efficientnet_lite_pytorch` pretrained weights) / augmentation / unlabeled pool
remain the suspects.

**CONTAMINATION NOTE**: editing `transforms.py` globally polluted the running
baseline AL run — each AL cycle spawns a fresh training subprocess that re-imports
`transforms.py`. hourglass+piano finished before the edit (clean); guitar cycle 1
trained under [-1,1] and was quarantined to
`results/miniimagenet_al/_CONTAMINATED_guitar_neg11/`. guitar/printer/orange must be
re-run clean. **Lesson for the real fix / future experiments: gate normalisation
behind a config field, never a global file edit, so concurrent runs don't collide.**

## DECISIVE: the regression is in TRAINING, not scoring (2026-07-06)
`cross_score_v1_native.py` rebuilds the v1 architecture
(`efficientnet_lite_pytorch`, installed `--no-deps`), loads v1's saved
`eval_model` weights (0 missing/unexpected keys), and scores all 65k images
through **v2's fitsbolt preprocessing + [0,1] ToTensor** (deterministic, no
training → no variance):

| v1 model | v1 reported AUROC | scored via v2 pipeline | AUPRC | top-1% |
|---|---|---|---|---|
| iter1 (cold 5/495) | 0.9614 | **0.9617** | 0.809 | 76.3% |
| iter3 (final) | 0.9702 | **0.9714** | 0.844 | 78.8% |

v2 scoring reproduces v1 to ±0.001. Therefore **v2 scoring/preprocessing is
faithful** and the whole v2 shortfall (0.81–0.88 vs 0.96) is in **training**.
Hyperparameters are byte-identical (v1 `config.toml`), so the training regression
is one of: (a) the **model swap** timm `tf_efficientnet_lite0.in1k` vs v1
`efficientnet_lite_pytorch` lite0 (commit `9f951f45`), or (b) a **FixMatch
training-loop change** in the v2 refactor. v2 training is also **high-variance**
(0.66–0.81 at fixed seed 42), unlike v1 (reliably ~0.96).

**Next isolating test**: run v2 training but with the v1 architecture (patch
`get_net_builder` to build the `efficientnet_lite_pytorch` lite0 for
"efficientnet-lite0"), one 5/495 cycle. ~0.96 → the timm model is the regression;
~0.81 → a v2 training-loop bug independent of the model.

## TRUE ROOT CAUSE: dropped regularisation in the timm build (2026-07-06, FINAL)
The backbone swap is the cause, but NOT via the weights. Decisive chain:
1. **Weights are byte-identical.** v1 (`efficientnet_lite0_pytorch_model`) pretrained
   weights vs timm `tf_efficientnet_lite0.in1k`: mean|Δ|=0.0, correlation=1.000. Same
   ImageNet checkpoint. (`efficientnet_lite0.ra_in1k` is a genuinely different run.)
2. **Architecture is numerically equivalent.** v1's *trained* weights loaded into the
   timm arch (positional map, 0 shape mismatches) score **0.9617** = v1 native. So
   eval-forward is identical.
3. Yet timm trains to ~0.81 (noisy) and v1 to ~0.967 (stable) from the *same* weights
   → the difference is **training-mode-only behaviour**. Three regularisation settings
   the v1 `efficientnet_lite_pytorch` model had but timm's default `create_model`
   build drops:

   | setting | v1 model | timm default | intended (cfg) |
   |---|---|---|---|
   | BN momentum | 0.01 | **0.1** | 0.01 (`cfg.bn_momentum = 1 - ema_m`) |
   | drop_connect / drop_path | 0.2 | **0.0** | — |
   | classifier dropout | 0.2 | **0.0** | — |

   **`cfg.bn_momentum` (0.01) is computed + validated but never applied to any BN
   layer** (`grep` finds no `.momentum =` on modules) — a genuine latent bug. v1's
   model defaulted to 0.01 so it silently matched; timm defaults to 0.1, so with
   batch size 16 the BN running stats move 10x too fast → the instability/variance and
   the ~0.81 plateau we measured. timm's build also omits EfficientNet's default
   stochastic-depth (0.2) and dropout (0.2).

**Fix (no weight swap / no git-LFS asset needed — weights are identical):** build the
timm model with `drop_path_rate=0.2, drop_rate=0.2` and apply `cfg.bn_momentum` to the
BN layers. Validation in flight (`results/fix_combined/` = all three; `fix_bnonly/` =
BN-momentum only, to find the minimal fix).

## (superseded) ROOT CAUSE CONFIRMED: the model swap, not normalisation (2026-07-06)
Isolating test — v1 architecture (`efficientnet_lite_pytorch` lite0) built inside
**v2's own training loop** (worktree patch of `get_net_builder`), plain [0,1],
hourglass 5/495 seed 42, 300 iters:

| setup | cycle 1 AUROC |
|---|---|
| v2 **timm** `tf_efficientnet_lite0.in1k`, [0,1] | 0.814 |
| v2 timm, [-1,1] | 0.809 |
| **v1 arch in v2 loop, [0,1]** | **0.9665** |
| v1 native (cross-score, deterministic) | 0.9617 |

Putting the v1 backbone in v2's training loop reproduces v1's ~0.96 immediately.
**Therefore the v2 training loop is correct; the whole regression is the backbone
swap** (commit `9f951f45`, `efficientnet_lite_pytorch` → timm
`tf_efficientnet_lite0.in1k`). The timm TF-ported lite0 weights transfer far worse
under this FixMatch setup. Input normalisation is only a partial mitigation for the
timm model (0.858 → 0.901 final avg) and never reaches 0.96.

**Both timm lite0 weight sets fail (2026-07-06):** `efficientnet_lite0.ra_in1k`
(timm-native RandAugment weights + their ImageNet mean/std) also gives cycle-1
0.7977 — as bad as the TF port. So it is not a "pick better timm weights" fix;
the timm `efficientnet_lite0` backbone underperforms `efficientnet_lite_pytorch`
(~0.967) regardless of which timm ImageNet weights are used.

**Weights vs architecture** (in flight, `convert_v1_to_timm.py`): v1 and timm lite0
state dicts align positionally (296 keys, 0 shape mismatches, distinct per-layer
shapes → unambiguous). Loading v1's trained weights into the timm arch and scoring
tests whether the timm *architecture* is equivalent (≈0.96 → arch fine, pretrained
init weights are the culprit; then convert v1's ImageNet weights into timm and
train).

**Fix direction (NEW):** the real fix is the model, not normalisation. Options:
1. Try timm-native `efficientnet_lite0.ra_in1k` (RandAugment-trained ImageNet
   weights, ImageNet mean/std) instead of the TF port — may match v1 while staying
   on timm.
2. Restore the `efficientnet_lite_pytorch` lite0 (proven 0.9665) — reintroduces the
   dependency #266 removed.
3. Deeper: diagnose why the TF-port lite0 underperforms (weights quality vs
   architecture/padding) — but (1) is the cheap first test.
The `get_normalisation_stats` + transform-normalisation work is still valid as
correct practice / a secondary improvement, but is NOT the primary fix.

## [-1,1] full-AL vs [0,1] vs v1 — CONSOLIDATED (2026-07-06)
Full 3-cycle AL, both configs, 300 iters/cycle, seed 42, hourglass + piano
(`results/miniimagenet_al_neg11/` = [-1,1]; `results/miniimagenet_al/` = [0,1]):

| class / cfg   | cycle1 | cycle2 | cycle3 |
|---------------|--------|--------|--------|
| hourglass **[-1,1]** | 0.809 | **0.924** | 0.887 |
| hourglass [0,1]      | 0.814 | 0.841 | 0.882 |
| piano **[-1,1]**     | 0.835 | **0.900** | **0.915** |
| piano [0,1]          | 0.861 | 0.872 | 0.833 |
| v1 (both)            | 0.961 | – | 0.970 |

Final-cycle avg: **[-1,1] = 0.901** vs **[0,1] = 0.858** vs **v1 = 0.966**.
[-1,1] (timm's documented Inception normalisation, mean/std 0.5) helps training
meaningfully and consistently at cycle 2 for both classes; cold cycle 1 is pure
noise (the earlier single [-1,1] run scored 0.66, this one 0.81 at identical
seed). So: **normalise to the pretrained model's native range**. Residual gap to
v1 (~0.06) + high variance remains → next: isolate whether it is the timm model
swap or a v2 training-loop change (run v2 training with the v1
`efficientnet_lite_pytorch` arch).

Verdict so far: scoring faithful (proven ±0.001); the regression is in TRAINING;
input normalisation is a real, defensible partial fix (final avg 0.858 → 0.901,
matches timm docs); a residual training gap remains under investigation.

## Ruled OUT (added 2026-07-06)
3. **Pretrained weights silently random** — RULED OUT. `timm.create_model(
   "tf_efficientnet_lite0.in1k", pretrained=True)` genuinely loads ImageNet weights
   here (timm 1.0.27): conv_stem std=0.77/absmax=3.7 (structured, not random init),
   bn1 running stats populated (running_var mean=6.7, not the default 1.0). The
   backbone is really pretrained.
   - Note: those BN running stats were computed for [-1,1]-normalised ImageNet
     inputs. v2 feeds [0,1]. With `bn_momentum=0.01` the running stats adapt slowly,
     but over 300 iters they do move toward the [0,1] activation distribution — so
     the [-1,1] full-AL test is the clean way to see if input range matters.

## Real fix (once direction is settled)
If normalisation (or any transform change) proves to help, implement it as a
config-driven option (e.g. `cfg.normalisation.input_normalisation_mean/std`,
defaulting to None = current [0,1] behaviour) read by the three transform builders,
resolved from the model's timm `pretrained_cfg` and persisted in the checkpoint so
prediction stays consistent. Do NOT hardcode into `transforms.py`.

## Leading hypotheses (superseded by root cause above)
The gap is in a part of the pipeline that changed in v2 with identical
hyperparameters:
1. **Image preprocessing / normalisation** — v2 uses **fitsbolt**
   `CONVERSION_ONLY`; v1 used the old loader. If fitsbolt produces different pixel
   values (resize interpolation, channel order, scaling) the pretrained net
   degrades. v2 prediction/weak/strong transforms end in `transforms.ToTensor()`
   → `[0,1]`, **no ImageNet mean/std Normalize** (`image_processing/transforms.py`);
   need to confirm whether v1 matched this (efficientnet-lite may intentionally
   use `[0,1]`, so this alone may be fine — verify).
2. **Augmentation** — weak/strong (RandAugment) implementation may have changed.
3. **Unlabeled training pool** — v1 `N_to_load=10000` (specific pool) vs v2
   `unlabeled_pool_cap`/hires sampling from all 65k.

## DECISIVE NEXT TEST (do first when resuming)
Score v1's own saved checkpoint through v2's prediction pipeline:
- v1 checkpoints (trained on identical 5/495):
  `.../benchmark_miniimagenet_anomaly57_*/model_iteration_0.pth` (also `_1`, `_2`).
- v1's own scores for each iteration are in
  `.../plots/plot_data/data_for_top_n_anomaly_detection_iter{1,2,3}.pkl`
  (dict: `scores`, `filenames`, `true_labels_df`, `anomaly_class`) — iter1 gives
  AUROC 0.961.
- **If v1's model scored via v2 pipeline ≈ 0.96** → v2 **training** is the
  regression. **If ≈ 0.81** → v2 **scoring/preprocessing** is the regression.

**Gotcha loading v1 .pth**: `torch.load(..., weights_only=False)` fails with
`ModuleNotFoundError: No module named 'anomaly_match.image_processing.NormalisationMethod'`
— the `NormalisationMethod` enum moved between v1 and v2. Register a compat alias
before loading, e.g.:
```python
import sys, anomaly_match.image_processing as ip
from anomaly_match.image_processing.normalisation_method import NormalisationMethod  # current location (verify)
sys.modules['anomaly_match.image_processing.NormalisationMethod'] = <module exposing the enum>
```
Then `state = torch.load(pth, weights_only=False)`; `state['eval_model']` is the
EMA state_dict (keys should match v2's FixMatch eval_model). Build v2 FixMatch,
`load_state_dict(state['eval_model'])`, then score with the v2 prediction path
(need a fitsbolt cfg — take it from a v2 checkpoint, or reconstruct via
`get_fitsbolt_config`).

Cheaper parallel check (no GPU): diff v1-era vs current
`anomaly_match/image_processing/` and `datasets/augmentation/` via git history
(v1 run date 2025-07-26) to see what changed in preprocessing/augmentation.

## Resume checklist
1. Let the AL run finish (or check `miniimagenet_al/summary.csv`); run
   `compare_v1_v2.py --v2-dir paper_scripts/v2/results/miniimagenet_al` for all
   classes → `paper_results_v2/comparisons/`.
2. Run the decisive cross-scoring test above.
3. If training regression: bisect preprocessing → augmentation → unlabeled pool.
4. GPU: run under `am` env (`conda run -n am`); active env is often `cutana_rust`.
