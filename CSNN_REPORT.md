# CSNN / SNN-MNIST Report

> Living log of experiments, fixes, and results.

## Summary

- Pipeline status: CSNN end-to-end now runs (train → label_map → counts → readouts) on CPU; GPU large runs were unstable previously due to stdout/tqdm BrokenPipe; fixes landed in tag `Csnn-eval-45`.

## Key fixes (chronological)

- Fixed CUDA device mismatch in label map by ensuring inputs/spikes are moved to the same device as the network runtime (`label_map.py`).
- Added CSNN (Conv2dConnection + LIF) skeleton and integrated into `experiment_runner.py` via `arch=csnn`.
- Added encoder output format `TBNCHW` to support Conv2dConnection input shape `[T,B,1,28,28]` (`encoders.py`).
- Updated counts collection to support conv monitor outputs `[T,B,C,H,W]` by flattening to `[T,B,N]` (`counts_readout.py`).
- Made eval device explicit (`evaluation.py` passes `device=cfg.torch_device()`), preventing accidental CUDA use.
- Made PCAwhiten optional (skip on degenerate covariance).
- Added progress logging to `status.json` + `live.log` in run folder; added early checkpoint after training (`model_after_train.pt`).
- Made tqdm robust against BrokenPipe/OSError in background runs.

## Results

### FC SNN (baseline)

- **Run** `20260421T182636Z_bfd61ae946ba` (status: ok)
  - Encoder: poisson (non-deterministic)
  - `n_hidden=200`, `top_k=3`, `T=200`, `N=12000`
  - Best readout: **TFIDF+MLP**
  - Accuracy: **0.3860**

### CSNN (1 conv layer)

- **Run** `20260422T114419Z_44c13caa9539` (status: ok)
  - Device: CPU
  - Conv1: `c1_out=16`, `kernel=5`, output shape 16×24×24 (= 9216 LIF)
  - Encoder: poisson, `TBNCHW`, deterministic
  - `T=20`, `N=200`, label_map calib=50, train_counts=200, test_counts=100
  - Best readout: **TFIDF+MLP**
  - Accuracy: **0.25**
  - Readouts: counts_zscore+Linear 0.22

### CSNN GPU large run

- **Run** `20260422T120049Z_13f871056c01` (status: failed)
  - Error: BrokenPipeError from tqdm (stdout pipe) during counts collection in eval
  - Fixed in tag `Csnn-eval-45` (tqdm safe + progress redirected to files)

### CSNN GPU resume-eval (from checkpoint)

- **Run** `20260423T154120Z_cd384ffafdd6` (status: ok)
  - Device: CUDA (GPU)
  - Resume checkpoint: `runs_csnn/20260423T042039Z_13f871056c01/model_after_train.pt`
  - Conv1: `c1_out=64`, `kernel=5`, output 64×24×24 (= 36,864 LIF)
  - Encoder: poisson, `TBNCHW`, deterministic, `rate_scale=0.011`, `rate_boost=3.0`
  - `T=100`, train_counts=2000, test_counts=500
  - Best readout: **counts_zscore+Linear**
  - Accuracy: **0.48**

### CSNN (Fashion-MNIST, grayscale)

- **Run** `20260428T154629Z_893fcd08a076` (status: ok)
  - Dataset: **Fashion-MNIST** (`dataset=fashion`), 28×28 grayscale
  - Device: CUDA (GPU)
  - Encoder: poisson, deterministic, `poisson_rate_scale=0.006`
  - CSNN: `c1_out=32`, `kernel=5` → 32×24×24 (= 18,432 LIF)
  - `T=100`, STDP train `N=5000`, label_map calib=1000, train_counts=5000, test_counts=5000
  - Best readout: **counts_zscore+Linear**
  - Accuracy: **0.7358**
  - Spikes/sample: **7254.914**
  - label_map coverage: **324 / 18432**

## Tuning log (what we swept)

### E/I inhibition strength (`ei_inh`) — short runs (N=500)

All: `arch=csnn`, CUDA, poisson deterministic, `poisson_rate_scale=0.006`, `T=100`, `N=500`, `n_calib=200`, counts eval 500/500.

- `ei_inh=160` → `runs_csnn/20260428T040917Z_4848b4ff2c41`
  - proportional_vote: **0.336**
  - spikes/sample: **3685.97**
  - label_map: **190 / 36864**

### `ei_inh` — short readout check (N=500)

All: `c1_out=32` (18,432 LIF), CUDA, `poisson_rate_scale=0.006`, `T=100`, `N=500`, `n_calib=200`, counts eval 500/500.

- `ei_inh=160` → `20260428T044521Z_0ba909110612` → counts_zscore+Linear **0.748**, label_map **149/18432**, spikes/sample **4286.63**
- `ei_inh=120` → `20260428T045027Z_c794d80e7657` → counts_zscore+Linear **0.760**, label_map **153/18432**, spikes/sample **4298.82**
- `ei_inh=90`  → `20260428T045541Z_29699e7b92d5` → counts_zscore+Linear **0.758**, label_map **143/18432**, spikes/sample **4307.41**

### Resume-only sweeps (weights frozen)

Baseline resume checkpoint: `runs_csnn/20260428T045027Z_c794d80e7657/model_after_train.pt`.
All: CUDA, `poisson_rate_scale=0.006`, `T=100`, `N=500`, `n_calib=1000`, counts eval 10k/10k.

- `ei_inh=120, theta_plus=0.05` → `20260428T053505Z_90b5cc6341da` → acc **0.8786**, label_map **165/18432**
- `theta_plus=0.02` (same) → `20260428T071545Z_70d9109a9ec4` → acc **0.8786**, label_map **165/18432** (no change)
- `ei_inh` sweep: 100/110/130/140 → `20260428T081809Z_...`, `082845Z_...`, `083924Z_...`, `084953Z_...`
  - all four: acc **0.8786**, label_map **165/18432** (no change)

### STDP scale-up (no resume)

All: CUDA, `poisson_rate_scale=0.006`, `T=100`, `ei_inh=120`, `theta_plus=0.05`, counts eval 10k/10k.

- `N=10000` → `20260428T091132Z_1e3d1d4cc831` → acc **0.8876**, label_map **214/18432**, spikes/sample **4122.89**
- `N=15000` → `20260428T113714Z_a80e83cb5cdd` → acc **0.8883**, label_map **219/18432**, spikes/sample **4146.94**
- `N=30000` → `20260428T112039Z_ecee7b2773c8` (aborted early)

### Homeostasis sweep (Fashion-MNIST, N=5000)

All: `arch=csnn`, `dataset=fashion`, CUDA, poisson deterministic (`poisson_rate_scale=0.006`), `T=100`, `N=5000`, `ei_inh=120`. Readout: **counts_zscore+Linear**.

- `homeo_target7500_gain0.10` → `20260428T210758Z_2e3af2ac48fe` → acc **0.7362**, spikes/sample **7255.08**, label_map **324/18432**
- `homeo_target8000_gain0.10` → `20260428T215319Z_1c29e7a91e99` → acc **0.7360**, spikes/sample **7254.61**, label_map **325/18432**
- `homeo_target7000_gain0.10` → `20260428T202300Z_cb788d17ef25` → acc **0.7358**, spikes/sample **7255.10**, label_map **324/18432**
- `homeo_target7000_gain0.15` → `20260428T193753Z_0c8a9f4085e7` → acc **0.7356**, spikes/sample **7254.80**, label_map **324/18432**
- `baseline_no_homeo` → `20260428T223758Z_91fef89ab729` → acc **0.7354**, spikes/sample **7254.37**, label_map **324/18432**

Takeaway: best homeostasis config is **+0.0008** over baseline (0.7362 vs 0.7354).

### Homeostasis check (Fashion-MNIST, N=15000)

Same settings as above, but `N=15000`. Two runs:

- `homeo_target7500_gain0.10` → `20260429T042820Z_d825bb17c324` → acc **0.7322**, spikes/sample **7222.06**, label_map **321/18432**
- `baseline_no_homeo` → `20260429T062336Z_a7ef25b0f838` → acc **0.7322**, spikes/sample **7222.13**, label_map **322/18432**

Takeaway: **no accuracy gain** from homeostasis at N=15000 in this setting (tie at 0.7322).

### `poisson_rate_scale` sweep (Fashion-MNIST, N=5000, homeo fixed) — 2026-04-30

All: `arch=csnn`, `dataset=fashion`, CUDA, poisson deterministic, `T=100`, `N=5000`, `n_calib=1000`, counts eval 5k/5k, `ei_inh=120`, `theta_plus=0.05`, `tau_theta=10000`, `w_norm_target=12.5`, homeo fixed (`target=7500`, `gain=0.10`, `lo/hi=6000/8500`, `warmup=500`). Readout: **counts_zscore+Linear**.

Artifacts:
- Log: `runs_csnn/sweep_rate_scale_homeo_fashion_N5000_20260430T035830Z.log`
- Summary JSONL: `runs_csnn/sweep_rate_scale_homeo_fashion_N5000_20260430T035830Z.summary.jsonl`

Runs:
- `rate_scale=0.004` → `20260430T035837Z_99a8a076553b` → acc **0.7284**, spikes/sample **6241.58**, label_map **351/18432**
- `rate_scale=0.005` → `20260430T044147Z_7780243a3c9c` → acc **0.7318**, spikes/sample **6548.72**, label_map **349/18432**
- `rate_scale=0.006` → `20260430T052625Z_2e3af2ac48fe` → acc **0.7356**, spikes/sample **7255.05**, label_map **324/18432**
- `rate_scale=0.007` → `20260430T061220Z_38a6b1f6d34d` → acc **0.7356**, spikes/sample **7853.49**, label_map **320/18432**
- `rate_scale=0.008` → `20260430T065653Z_577281631a58` → acc **0.7372** (best), spikes/sample **8137.38**, label_map **306/18432**

Takeaway: within **0.004→0.008** the accuracy trends slightly upward with `poisson_rate_scale`, at the cost of higher spike counts.

## Current configuration notes

- CSNN Conv1 neuron count (LIF): `c1_out * (28 - k + 1 + 2*pad)^2`.
  - Example: c1_out=64, k=5 → output 24×24 → 64×24×24 = 36,864 LIF.

## Tags / versions

- `Csnn-eval-45` → commit `7eee674` (CSNN eval pipeline + robust progress logging)

## Next steps (planned)

- Add Diehl & Cook style competition (E/I loop or strong lateral inhibition in conv maps).
- Implement label_map-based classifier (vote by assigned neuron spikes) as an alternative to external readout.
- Add periodic intermediate eval on a small held-out set to track best_acc_so_far during training.
