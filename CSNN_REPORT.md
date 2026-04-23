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

## Current configuration notes

- CSNN Conv1 neuron count (LIF): `c1_out * (28 - k + 1 + 2*pad)^2`.
  - Example: c1_out=64, k=5 → output 24×24 → 64×24×24 = 36,864 LIF.

## Tags / versions

- `Csnn-eval-45` → commit `7eee674` (CSNN eval pipeline + robust progress logging)

## Next steps (planned)

- Add Diehl & Cook style competition (E/I loop or strong lateral inhibition in conv maps).
- Implement label_map-based classifier (vote by assigned neuron spikes) as an alternative to external readout.
- Add periodic intermediate eval on a small held-out set to track best_acc_so_far during training.
