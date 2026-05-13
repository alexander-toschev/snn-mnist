# CSNN / CIFAR100:20 — experiment log

Цель (гейт 1): 2-layer должен выйти хотя бы в **0.25–0.35** быстро на cifar100:20.

---

## 2026-05-01 (CIFAR100:20 RGB32)

### 1-layer baseline / competition (pad2)
- **pad2 + EI** (wta=off, ei=on):
  - 20260501T085357Z_747e1b5e25d7 → acc=**0.172** (counts_zscore+Linear), assigned=392/32768
- **pad2 + WTA** (wta=on, ei=off):
  - 20260501T094221Z_6ad14ae06bd4 → acc=**0.165**, assigned=988/32768

Вывод: EI даёт чуть лучше acc (~0.17), WTA даёт больше «победителей», но acc не растёт.

### EI inh sweep (pad2, N=5000)
- ei_inh=80  → 20260501T121527Z_2d084c83e12f → acc=0.171
- ei_inh=120 → 20260501T130827Z_747e1b5e25d7 → acc=**0.173** (лучшее в этом свипе)
- ei_inh=160 → 20260501T135622Z_990fb177c979 → acc=0.1715

Вывод: в диапазоне 80–160 метрика почти не меняется.

### Capacity (1-layer)
- c1_out=64 (pad2+EI):
  - 20260501T144835Z_91ccc96a8c54 → acc=**0.159**

Вывод: простое увеличение c1_out ухудшило качество.

### 2-layer greedy
- 2-layer greedy + WTA (N=5000, greedy_n1=2500):
  - FAIL: 20260501T162706Z_358915b06a96 → `zalipe detected... at i=1000`
  - OK:   20260501T163648Z_358915b06a96 → acc=**0.1965**, assigned=640/16384

Вывод: 2-layer даёт jump (~0.17 → ~0.20), но до гейта 0.25–0.35 ещё далеко.

---

## 2026-05-02

### Перезапуск 2-layer greedy+WTA
- 20260502T070527Z_358915b06a96 → FAIL на eval: `IndexError: list index out of range`
  - причина: в eval стоял n_train_counts=60000, а cifar100:20 train_len=10000 → DataLoader лез за пределы.

### Fix
- counts_readout.collect_counts_plus_fast: добавлен cap `n_samples = min(n_samples, len(ds))`.

### Eval-only resume (проверка фикса + метрика)
- 20260502T080803Z_4ba4d8c7497f (resume от model_after_train.pt) → OK
  - acc=**0.187** (counts_zscore+Linear), assigned=605/16384

### 2-layer greedy + EI (wta=off, ei=on)
- 20260502T095644Z_221089919b62 → OK, но сеть практически молчит
  - acc=**0.050** (counts_zscore+Linear)
  - spikes/sample≈**91.7**
  - assigned=**0/16384**
  - Вывод: текущие параметры EI (по умолчанию `ei_inh=120`) для 2-layer в этой реализации «душат» активность.

### 2-layer greedy + WTA+EI (до фикса логики)
- 20260502T111112Z_81024f266893 → ABORTED вручную (по симптомам сеть тоже была задавлена EI)

### 2-layer greedy + WTA+EI (ei_inh=40)
- 20260502T114114Z_f6da460ee7c2 → FAIL: `low activity: spikes_win_mean=37.150 < 1000.000 at i=1000 (in_spikes=945)`

### Fix: EI в 2-layer
- В `csnn_mnist_net.py` добавлен параметр `ei_inh_mult_2layer` (default **0.01**): при `c2_out>0` ингибиция масштабируется, чтобы не душить сеть.
- (Ранее) WTA-after-EI добавлен; (ранее) activity-guard добавлен в раннер.

### Fix (bug): EI раньше занулял Conv1 spikes → Conv2 голодал
- В ветке `ei_enable` убрали пересчёт `conv_lif.s = conv_lif.v >= thresh` (он приводил к нулевым Conv2 spikes).
- Теперь EI влияет через `conv_lif.v -= inh * i_s`, а WTA (если включён) применяется поверх уже рассчитанных spikes.

---

## Что делаем дальше (зафиксировано)

База: **2-layer greedy + WTA** (как минимум конфиг уровня 20260501T163648Z_358915b06a96).

Следующие эксперименты (меняем только competition):
1) 2-layer greedy + **EI** (wta=off, ei=on), N=5000
2) 2-layer greedy + **WTA+EI** (wta=on, ei=on), N=5000

Критерий: ищем рост относительно 0.1965 и приближаемся к гейту 0.25–0.35.

- 2026-05-02 20260502T095644Z_221089919b62 | wta=0 ei=1 greedy=1 N=5000 rate=0.008 | ok best_acc=0.05 assigned=0/16384
- 2026-05-02 20260502T133156Z_9c06d933fe14 | wta=1 ei=1 greedy=1 N=5000 rate=0.008 time=100 | ei_exc=22.5 ei_inh=120 mult2=0.01 | FAILED: low activity (spikes_win_mean=27.38 < 1000 at i=1000, in_spikes=945)

### 2-layer greedy + EI (fixed) — STALE
- 20260502T160528Z_fdebdc16466c → FAIL: stale (process died / no updates after last status.json)


### 2026-05-02 — run 20260502T160528Z_fdebdc16466c (csnn, cifar100:20)
- flags: wta=False, ei=True, ei_inh_mult_2layer=0.01
- result: FAILED (stage=stale; status=failed) at 63.5% (i=3175/5000)
- error: RuntimeError: stale run: no running process found; last updated_at=2026-05-02T16:39:28Z
- updated_at: 2026-05-02T16:39:28Z; finished_at: 2026-05-02T18:15:52Z
- run_id: 20260502T160528Z_fdebdc16466c

### 2-layer greedy + EI (fixed retry) — OK
- 20260502T181648Z_fdebdc16466c (csnn, cifar100:20)
  - flags: greedy=1 wta=0 ei=1 | ei_inh=120 mult2=0.01 | N=5000 rate=0.008 time=100 (poisson_deterministic=1) | w_norm=1 target=12.5 | adapt_thresh=1
  - best_acc=**0.1875** (counts_zscore+Linear)
  - assigned=**686/16384**
  - spikes/sample≈**43332**
  - finished_at=2026-05-02T19:29:11Z

### 20260502T200107Z_9c06d933fe14 (finished 2026-05-02T21:21:48Z)
- wta: true; ei: true; ei_inh: 120; ei_inh_mult_2layer: 0.01
- spikes_per_sample: 43345.9618
- assigned: 686/16384
- best_acc: 0.186 (counts_zscore+Linear)
- error: null


---
### 20260502T133156Z_9c06d933fe14 — failed
- created_at: 2026-05-02T13:31:56Z
- finished_at: 2026-05-02T13:42:44Z
- dataset: cifar100:20
- arch: csnn
- config_hash: 9c06d933fe14
- flags: WTA, EI, EI_inh_mult_2layer=0.01, adapt_thresh, w_norm, greedy(n1=2500), poisson_deterministic
- error_type: RuntimeError
- error: low activity: spikes_win_mean=27.380 < 1000.000 at i=1000 (in_spikes=945)
- logged_at: 2026-05-03T02:57:31.468Z


## 2026-05-03

### Stabilize: 2-layer greedy + WTA + EI + homeostasis (N=2000)
- 20260503T055709Z_624ca258d83e (finished 2026-05-03T06:28:44Z)
  - cfg: greedy=1 (n1=1000) wta=1 ei=1 mult2=0.01 homeo=1 | time=100 | poisson_rate_scale(base)=0.006
  - homeo corridor: lo=20000 hi=30000 target=25000 (gain=0.5, ema_alpha=0.02)
  - best_acc=**0.163** (counts_zscore+Linear)
  - spikes/sample=**26680.45**
  - assigned=**637/16384**
  - note: spikes заметно ниже, чем у прошлых прогонов (~42–44k), но acc ниже (возможно из-за N=2000)

### Stabilize: 2-layer greedy + WTA + EI + homeostasis (N=5000)
- 20260503T064315Z_4ea08004de82 (finished 2026-05-03T07:50:41Z)
  - cfg: greedy=1 (n1=2500) wta=1 ei=1 mult2=0.01 homeo=1 | time=100 | poisson_rate_scale(base)=0.006
  - homeo corridor: lo=20000 hi=30000 target=25000 (gain=0.5, ema_alpha=0.02)
  - best_acc=**0.1645** (counts_zscore+Linear)
  - spikes/sample=**24622.96**
  - assigned=**643/16384**
  - note: стабилизация по spikes работает, но качество просело относительно baseline WTA (0.1965 при ~44.6k spikes/sample)

### Stabilize: 2-layer greedy + WTA + EI + homeostasis, corridor 30k–45k (N=5000)
- TRAIN run: 20260503T081520Z_a4531140e5a2 → дошёл до конца, но завис на stage=readout_probe (нет summary/model.pt)
  - activity at i=5000: spikes_win_mean≈**34698**, spikes_win_max≈**89920**, homeo_rate_scale=0.006 (in_band)
- RESUME+EVAL от model_after_train.pt:
  - 20260503T100934Z_419c50aec6da (finished 2026-05-03T10:24:57Z)
    - best_acc=**0.1845** (counts_zscore+Linear)
    - assigned=**602/16384**
    - note: spikes/sample в summary=0 (т.к. resume без train); для активности см. activity.jsonl в train-run

### WTA+EI (NO homeostasis) — fixed rate_scale=0.0035 (N=5000)
- 20260503T115413Z_e03ac87c4aa8 (finished 2026-05-03T13:02:06Z)
  - best_acc=**0.1715** (counts_zscore+Linear)
  - spikes/sample=**24519.16**
  - assigned=**633/16384**
  - note: почти те же spikes, что и при homeo(20–30k), но acc чуть выше (0.1715 vs 0.1645)

### WTA-only (NO EI, NO homeostasis) — rate_scale=0.006 (N=5000)
- 20260503T183811Z_9fa5cca253da (finished 2026-05-03T19:30:28Z)
  - best_acc=**0.1925** (counts_zscore+Linear)
  - spikes/sample=**36650.65**
  - assigned=**633/16384**
  - note: очень близко к baseline 0.1965, при этом spikes ниже, чем у baseline (~44635)

---
### CSNN run 20260502T133156Z_9c06d933fe14 — FAILED (WTA+EI mult2layer001)
- dataset: cifar100:20 | device: cuda | arch: csnn
- created_at: 2026-05-02T13:31:56Z | finished_at: 2026-05-02T13:42:44Z
- progress: i=1000/5000 (20%)
- flags: wta=on; ei=on (exc=22.5, inh=120.0, inh_mult_2layer=0.01); adapt_thresh=on; greedy=on (n1=2500); w_norm=on (target=12.5)
- encoder: poisson (deterministic, rate_scale=0.008, rate_boost=1.0) | time=100
- error: low activity: spikes_win_mean=27.380 < 1000.000 at i=1000 (in_spikes=945)

---
### CSNN run 20260502T160528Z_fdebdc16466c — FAILED (EI fixed)
- dataset: cifar100:20 | device: cuda | arch: csnn
- created_at: 2026-05-02T16:05:28Z | finished_at: 2026-05-02T18:15:52Z
- progress: i=3175/5000 (63.5%)
- flags: wta=off; ei=on; ei_inh_mult_2layer=0.01
- error: RuntimeError: stale run (no running process found; last updated_at=2026-05-02T16:39:28Z)

### 2026-05-04 — sweep WTA-only (2-layer greedy, no EI, no homeo)
- run_id: 20260504T093816Z_c7b09ead45d5
- rate_scale: 0.007 | budgets: n_calib=1000 n_train_counts=5000 n_test_counts=2000
- status=ok stage=None acc=0.1945 spikes_per_sample=40788.5304 assigned=636/16384
- log: runs_csnn/sweep_wta_only_rate_scale_0.007_20260504T093812Z.log

### 2026-05-04 — sweep WTA-only (2-layer greedy, no EI, no homeo)
- run_id: 20260504T103159Z_495506eec83e
- rate_scale: 0.008 | budgets: n_calib=1000 n_train_counts=5000 n_test_counts=2000
- status=ok stage=None acc=0.195 spikes_per_sample=44627.052 assigned=641/16384
- log: runs_csnn/sweep_wta_only_rate_scale_0.008_20260504T093812Z.log

### 2026-05-04 — sweep WTA-only (2-layer greedy, no EI, no homeo)
- run_id: 20260504T112530Z_f9f7224d9ee7
- rate_scale: 0.009 | budgets: n_calib=1000 n_train_counts=5000 n_test_counts=2000
- status=ok stage=None acc=0.192 spikes_per_sample=48248.595 assigned=641/16384
- log: runs_csnn/sweep_wta_only_rate_scale_0.009_20260504T093812Z.log

### 2026-05-04 — sweep WTA-only (2-layer greedy, no EI, no homeo)
- run_id: 20260504T121922Z_5b7f226bbcd5
- rate_scale: 0.010 | budgets: n_calib=1000 n_train_counts=5000 n_test_counts=2000
- status=ok stage=None acc=0.193 spikes_per_sample=51630.691 assigned=645/16384
- log: runs_csnn/sweep_wta_only_rate_scale_0.010_20260504T093812Z.log


## 2026-05-04 18:24 UTC — run 20260502T160528Z_fdebdc16466c (CSNN, EI fixed)
- outcome: ERROR (status=failed)
- progress: 63.5%
- flags: wta=off, ei=on, ei_inh_mult_2layer=0.01

---
### 2026-05-04 — 3-layer (Conv1→Conv2→Conv3) greedy + WTA-only (no EI, no homeo)
- run_id: 20260504T194946Z_6c1b8c28e5c9 (finished 2026-05-04T21:02:36Z)
  - arch: c1=32 k5 s1 p2 | c2=64 k3 s2 p1 | c3=128 k3 s2 p1
  - greedy: n1=2000, n2=2000
  - encoder: poisson (deterministic), rate_scale=0.008, time=100
  - best_acc=**0.173** (counts_zscore+Linear)
  - spikes/sample=**58709.68**
  - assigned=**164/8192**

## 2026-05-06 — eval-only resume (3-layer greedy + local_inhib + homeo, top_k=20)
- исходный train run: 20260505T183309Z_60a47286351e
- eval run: 20260506T052927Z_ea635b5e466f
  - budgets: n_calib=1000, n_train_counts=60000, n_test_counts=10000
  - best_acc=**0.137** (counts_zscore+Linear)
  - assigned=**1006/8192**

## 2026-05-06 — 2-layer greedy + local_inhib (no WTA, no EI, no homeo)
- run_id: 20260506T082140Z_7c1ea8971373 (finished 2026-05-06T09:11:07Z)
  - cfg: greedy=1 (n1=2500) local_inhib=1 (strength=0.85) wta=0 | rate_scale=0.008 | time=100 | adapt_thresh=1 | w_norm=1 (target=12.5)
  - best_acc=**0.1955** (counts_zscore+Linear)
  - spikes/sample=**44626.86**
  - assigned=**639/16384**

## 2026-05-06 — sweep: local_inhib_strength (rate_scale=0.008, time=100)
- 0.75 → 20260506T094835Z_f5c04bfe1fc9 → acc=**0.1955**, spikes/sample=44630.41, assigned=641/16384
- 0.85 → 20260506T104118Z_7c1ea8971373 → acc=**0.1955**, spikes/sample=44630.19, assigned=640/16384
- 0.95 → (resume+eval) 20260506T130605Z_f89312662a47 → acc=**0.176**, assigned=557/16384

## 2026-05-06 — sweep: poisson_rate_scale (local_inhib=0.85, time=100)
- 0.007 → 20260506T132044Z_956ee7406f0b → acc=**0.1965**, spikes/sample=40774.61, assigned=637/16384
- 0.008 → 20260506T141148Z_7c1ea8971373 → acc=**0.1935**, spikes/sample=44615.37, assigned=639/16384
- 0.009 → 20260506T150224Z_4baccc9ea0a5 → acc=**0.1895**, spikes/sample=48247.92, assigned=641/16384

## 2026-05-07 — resume+eval (from stalled label_map run)
- resume_from: 20260506T113211Z_8c12de619250 (model_after_train.pt)
- eval run: 20260507T083853Z_f89312662a47 → acc=**0.192** (counts_zscore+Linear), assigned=558/16384
## 2026-05-07 — A/B: grayscale input (gray3) vs RGB baseline
- cfg: input_mode=gray3 (grayscale replicated to 3 channels), rate_scale=0.007, local_inhib=0.85, time=100, N=5000
- run_id: 20260507T094851Z_869414e168b6 → acc=**0.204** (counts_zscore+Linear), spikes/sample=41288.75, assigned=662/16384

⚠️ NOTE: этот acc=0.204 был посчитан с несостыкованным препроцессингом (train=gray3, но label_map/eval по умолчанию были RGB).
После фикса (label_map/eval тоже используют input_mode) корректный resume+eval:
- run_id: 20260507T111553Z_bd63b25d2002 → acc=**0.177** (counts_zscore+Linear), assigned=569/16384


## ЕДИНЫЙ ЖУРНАЛ
Дополнительная сводка/хронология по всем датасетам ведётся в: `EXPERIMENTS-HISTORY.md`.


---
### 20260502T200107Z_9c06d933fe14 (finished_at: 2026-05-02T21:21:48Z)
- wta=1 ei=1 ei_inh=120.0 ei_inh_mult_2layer=0.01
- spikes_per_sample=43345.9618
- assigned=686
- best_acc=0.186
- error=None

---
### CSNN run 20260502T181648Z_fdebdc16466c — OK (EI fixed, retry)
- dataset: cifar100:20 | device: cuda | arch: csnn
- created_at: 2026-05-02T18:16:48Z | finished_at: 2026-05-02T19:29:11Z
- flags: greedy=on (n1=2500); wta=off; ei=on (inh=120.0, inh_mult_2layer=0.01); adapt_thresh=on; w_norm=on (target=12.5); homeo=off
- encoder: poisson (deterministic, rate_scale=0.008, rate_boost=1.0) | time=100 | N=5000
- best_acc=**0.1875** (counts_zscore+Linear)
- assigned=**686/16384** | spikes/sample=**43332.19**
- error: None
- 2026-05-06 19:55 UTC run_id=20260502T095644Z_221089919b62 N=5000 → ok, best_acc=0.05

---
### CSNN run 20260502T133156Z_9c06d933fe14 — FAILED (appended 2026-05-09T00:35:54Z)
- dataset: cifar100:20 | device: cuda | arch: csnn
- created_at: 2026-05-02T13:31:56Z | finished_at: 2026-05-02T13:42:44Z
- flags: wta=on; ei=on; ei_exc=22.5; ei_inh=120.0; ei_inh_mult_2layer=0.01; greedy=on; n1=2500; adapt_thresh=on; w_norm=on; w_norm_target=12.5; homeo=off
- encoder: poisson (deterministic=True, rate_scale=0.008, rate_boost=1.0) | time=100 | N=5000
- progress: i=1000/5000 (20%)
- error: RuntimeError: low activity: spikes_win_mean=27.380 < 1000.000 at i=1000 (in_spikes=945)


---
### Run 20260502T133156Z_9c06d933fe14 (2026-05-10 01:14:34 +0300)
- **status:** failed
- **created_at:** 2026-05-02T13:31:56Z
- **finished_at:** 2026-05-02T13:42:44Z
- **run_dir:** `runs_csnn/20260502T133156Z_9c06d933fe14`
- **config_hash:** `9c06d933fe14`
- **error:** low activity: spikes_win_mean=27.380 < 1000.000 at i=1000 (in_spikes=945)


## Run 20260502T133156Z_9c06d933fe14
- time: 2026-05-10T01:58:47+03:00
- status: failed
- error: low activity: spikes_win_mean=27.380 < 1000.000 at i=1000 (in_spikes=945)
- flags: `{"N": 5000, "activity_check_after": 1000, "activity_min_spikes_win_mean": 1000.0, "adapt_thresh_enable": true, "arch": "csnn", "c1_kernel": 5, "c1_out": 32, "c1_pad": 2, "c1_stride": 1, "c2_kernel": 3, "c2_out": 64, "c2_pad": 1, "c2_stride": 2, "dataset": "cifar100:20", "device": "cuda", "ei_enable": true, "ei_exc": 22.5, "ei_inh": 120.0, "ei_inh_mult_2layer": 0.01, "encoder": "poisson", "encoder_rate_boost": 1.0, "greedy_enable": true, "greedy_n1": 2500, "homeo_enable": false, "input_channels": 3, "input_h": 32, "input_w": 32, "poisson_deterministic": true, "poisson_rate_scale": 0.008, "tau_theta": 10000, "theta_plus": 0.05, "time": 100, "w_norm_enable": true, "w_norm_target": 12.5, "wta_enable": true}`

### 2026-05-02 16:05:28 UTC — 20260502T160528Z_fdebdc16466c
- dataset: `cifar100:20`
- flags: wta=True, ei=True, ei_inh_mult_2layer=0.01
- status: **failed** (stage=stale)
- progress: 63.5% (i=3175/5000)
- error: `stale run: no running process found; last updated_at=2026-05-02T16:39:28Z`
- updated_at: 2026-05-02T16:39:28Z
- finished_at: 2026-05-02T18:15:52Z


---
## 2026-05-09 — Diverse10 (cifar100:0,1,2,3,5,8,13,14,17,19) — цель: поднять acc

Все ниже: input_mode=rgb, local_inhib=0.85, greedy_n1=2500, EI=off, WTA=off, adapt_thresh=on, w_norm_target=12.5, N=5000.

- 3-layer (c3_out=128), T=100, rate_scale=0.007 → run_id=20260509T085735Z_121c0577da30 → best_readout_acc=**0.297** | spikes/sample=53625 | assigned=175/8192
- 3-layer (c3_out=128), T=200, rate_scale=0.007 → run_id=20260509T102615Z_b779aa59200d → best_readout_acc=**0.314** | spikes/sample=145046 | assigned=155/8192
- 2-layer, T=300, rate_scale=0.007 → run_id=20260509T112827Z_a45fd48d8259 → best_readout_acc=**0.322** | spikes/sample=170008 | assigned=640/16384
- 3-layer (c3_out=256), T=100, rate_scale=0.007 → run_id=20260509T125440Z_5c15461cec09 → best_readout_acc=**0.297** | spikes/sample=107253 | assigned=175/16384

Компромиссы по spike-бюджету:
- 2-layer, T=200, rate_scale=0.0035 → run_id=20260509T164028Z_a817e778353e → best_readout_acc=**0.313** | spikes/sample=63090 | assigned=641/16384
- 2-layer, T=300, rate_scale=0.00233 → run_id=20260509T183647Z_8db3d1e00da6 → best_readout_acc=**0.310** | spikes/sample=76720 | assigned=632/16384

Сбой:
- run_id=20260509T144624Z_a817e778353e (T=200, rate_scale=0.0035) → завис/не дописал summary (остановился ~4.48%, i=224).

Вывод: 2-layer даёт существенно более «богатое» распределение winner'ов и assigned, чем 3-layer, и выглядит лучшей базой для роста точности.


---
## 2026-05-10 — План: Diverse10 accuracy push (архитектура → N → readout)

### Этап A (N=5000, T=200, rate_scale=0.0035): sweep ёмкости
Запускаем 4 прогона (2-layer):
1) baseline: c1_out=32, c2_out=64, greedy_n1=2500
2) widen:   c1_out=64, c2_out=128, greedy_n1=2500
3) neurons: c1_out=32, c2_out=64, greedy_n1=5000
4) both:    c1_out=64, c2_out=128, greedy_n1=5000

Критерий выбора: best_readout_acc (counts_zscore+Linear) + sanity по assigned/winners_unique.

### Этап B (увеличить N на лучшей конфигурации)
Повторить лучшую конфигурацию на N=20000 (и при необходимости выше).

### Этап C (readout)
На лучших чекпойнтах прогнать MLP readout и сравнить с Linear.

### Результаты Этап A (N=5000)
(общее: input_mode=rgb, local_inhib=0.85, EI=off, WTA=off, adapt_thresh=on, w_norm_target=12.5)

1) baseline (c1_out=32, c2_out=64, greedy_n1=2500)
- run_id=20260510T051713Z_a817e778353e → best_readout_acc=**0.309** | spikes/sample=63079.55 | assigned=636/16384 | winners_unique=237

2) widen (c1_out=64, c2_out=128, greedy_n1=2500)
- (попытка) run_id=20260510T065532Z_57ddeb24feac → ЗАВИС на i=896/5000 (17.92%), stage=collect_train_counts, updated_at=2026-05-10T08:36:42Z (summary.json не создан)
- run_id=20260510T134240Z_57ddeb24feac → best_readout_acc=**0.326** | spikes/sample=126231.17 | assigned=638/32768 | winners_unique=236

3) neurons (c1_out=32, c2_out=64, greedy_n1=5000)
- run_id=20260510T155529Z_7207abc2991f → best_readout_acc=**0.314** | spikes/sample=63113.16 | assigned=637/16384 | winners_unique=235

4) both (c1_out=64, c2_out=128, greedy_n1=5000)
- run_id=20260510T173518Z_ace2c9f33fe6 → best_readout_acc=**0.322** | spikes/sample=126210.94 | assigned=632/32768 | winners_unique=238

Итог Этап A: лучшая конфигурация = **widen (64/128, greedy_n1=2500)** → **0.326**.

### Результат Этап B (N=20000) на лучшей конфигурации
- widen (c1_out=64, c2_out=128, greedy_n1=2500), N=20000:
  - run_id=20260510T192621Z_6b08c66d6220 → best_readout_acc=**0.325** | spikes/sample=126249.65 | assigned=634/32768 | winners_unique=236



### 2026-05-02T20:01:07Z (20260502T200107Z_9c06d933fe14)
- wta: True
- ei: True
- ei_inh: 120.0
- ei_inh_mult_2layer: 0.01
- spikes_per_sample: 43345.9618
- assigned: 686/16384
- best_acc: 0.186
- error: None

### 20260502T133156Z_9c06d933fe14 — failed
- created_at: 2026-05-02T13:31:56Z
- finished_at: 2026-05-02T13:42:44Z
- config_hash: 9c06d933fe14
- run_dir: runs_csnn/20260502T133156Z_9c06d933fe14
- flags: arch="csnn" dataset="cifar100:20" device="cuda" time=100 N=5000 encoder="poisson" poisson_deterministic=true poisson_rate_scale=0.008 encoder_rate_boost=1 wta_enable=true ei_enable=true ei_exc=22.5 ei_inh=120 ei_inh_mult_2layer=0.01 adapt_thresh_enable=true greedy_enable=true greedy_n1=2500 w_norm_enable=true w_norm_target=12.5 c1_out=32 c1_kernel=5 c1_stride=1 c1_pad=2 c2_out=64 c2_kernel=3 c2_stride=2 c2_pad=1 activity_check_after=1000 activity_min_spikes_win_mean=1000
- error: (RuntimeError) low activity: spikes_win_mean=27.380 < 1000.000 at i=1000 (in_spikes=945)

---

## 2026-05-12/13 — Diverse10 (cifar100:0,1,2,3,5,8,13,14,17,19), WIDE 2-layer (c1=64,c2=128), local_inhib=0.85

### TRAIN (T=300, rate_scale=0.0035, greedy_n1=2500, N=20000)
- run_id=20260512T174140Z_de37f37d301e → status=ok (train+checkpoint+label_map), **readout не считался** (best_readout=null)
  - spikes/sample≈209390.25 | winners_unique=230
  - label_map assigned=628/32768

### RESUME+EVAL (от 20260512T174140Z_de37f37d301e/model_after_train.pt)
- Linear-only (без TFIDF):
  - run_id=20260513T051258Z_027f24383bb5 → best_readout_acc=**0.319** (counts_zscore+Linear)
  - label_map assigned=562/32768

- TFIDF+MLP (env `SNN_READOUT_TFIDF_MLP=1`, skip label_map, memmap, batch_size=32):
  - run_id=20260513T054352Z_027f24383bb5 → **best_readout_acc=0.3790 (TFIDF+MLP)**
    - counts_zscore+Linear=0.318

Примечание: фикс для memmap readout (torch.log1p на memmap) и env-оверрайд batch_size добавлены в `evaluation.py`.
