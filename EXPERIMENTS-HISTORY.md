# История экспериментов (единый журнал)

Назначение: единый, пополняемый журнал для подготовки статьи/отчёта.
Формат записи: дата (UTC), dataset, архитектура/режим, ключевые параметры, метрики (acc/spikes/assigned), вывод.

---

## 1) MNIST / Fashion-MNIST

Источник деталей: `CSNN_REPORT.md`.

### 1.1 MNIST — базовая полносвязная SNN
- run_id: 20260421T182636Z_bfd61ae946ba
  - модель: FC SNN (poisson), n_hidden=200, top_k=3, T=200, N=12000
  - лучшее считывание: TFIDF+MLP
  - accuracy: 0.3860

### 1.2 MNIST — CSNN (1 conv слой, STDP) — проверка пайплайна
- run_id: 20260422T114419Z_44c13caa9539
  - Conv1: c1_out=16, k=5 → 16×24×24 (=9216 нейронов)
  - T=20, N=200
  - accuracy: 0.25 (TFIDF+MLP), counts_zscore+Linear 0.22

### 1.3 MNIST — CSNN GPU (resume-eval)
- run_id: 20260423T154120Z_cd384ffafdd6
  - Conv1: c1_out=64, k=5 → 64×24×24 (=36864 нейронов)
  - encoder: poisson deterministic, rate_scale=0.011, rate_boost=3.0, T=100
  - best_readout: counts_zscore+Linear
  - accuracy: 0.48

### 1.4 Fashion-MNIST — CSNN (Conv1+STDP) — основной результат
- run_id: 20260428T154629Z_893fcd08a076
  - dataset: fashion, 28×28, grayscale
  - Conv1: c1_out=32, k=5 → 32×24×24 (=18432 нейронов)
  - encoder: poisson deterministic, rate_scale=0.006, T=100
  - STDP train: N=5000; label_map calib=1000; counts eval: 5000/5000
  - best_readout: counts_zscore+Linear
  - accuracy: 0.7358
  - spikes/sample: 7254.914
  - label_map coverage: 324/18432

### 1.5 Fashion-MNIST — масштабирование STDP (без resume)
- run_id: 20260428T091132Z_1e3d1d4cc831
  - N=10000 → accuracy 0.8876; label_map 214/18432; spikes/sample 4122.89
- run_id: 20260428T113714Z_a80e83cb5cdd
  - N=15000 → accuracy 0.8883; label_map 219/18432; spikes/sample 4146.94

---

## 2) CIFAR100:20 — CSNN (2–3 conv слоя, STDP)

Источник деталей (2-layer свипы/база): `EXPERIMENTS-CSNN-CIFAR10020.md`.

### 2.1 CIFAR100:20 — 2-layer baseline (WTA-only, greedy)
- ориентир лучшего режима (по журналу): accuracy ≈ 0.194–0.196 (counts_zscore+Linear) при rate_scale 0.007–0.008

### 2.2 CIFAR100:20 — 3-layer (Conv1→Conv2→Conv3), greedy + WTA-only
- run_id: 20260504T194946Z_6c1b8c28e5c9 (finished 2026-05-04T21:02:36Z)
  - c1=32 k5 s1 p2 | c2=64 k3 s2 p1 | c3=128 k3 s2 p1
  - greedy: n1=2000, n2=2000; encoder poisson deterministic; rate_scale=0.008; time=100
  - accuracy: 0.173 (counts_zscore+Linear)
  - spikes/sample: 58709.68
  - assigned: 164/8192
  - вывод: глубина ухудшила качество относительно 2-layer; высокая активность, низкая специализация (assigned).

---

## 3) Диагностические прогоны (CIFAR100:20, 3-layer) — 2026-05-05

Общий контекст: добавлены (а) применение competition/adapt_thresh ко всем conv слоям, (б) per-layer логирование активности, (в) метрика распределения нейронов по классам (neurons_per_class).

### 3.1 3-layer, WTA-only, без homeostasis (пер-layer логирование)
- run_id: 20260505T041502Z_846e7f33ee14
  - N=1200, greedy_n1=300, greedy_n2=300, rate_scale=0.008
  - spikes/sample: 61226.67
  - assigned: 145/8192
  - neurons_per_class: mean=7.25, min=0, max=19, cv=0.750

### 3.2 3-layer, homeostasis (target=20000; 15000–25000)
- run_id: 20260505T080432Z_a8d1d58c82b5
  - spikes/sample: 30047.25
  - assigned: 147/8192
  - neurons_per_class: mean=7.35, min=0, max=17, cv=0.757
  - вывод: homeostasis стабилизирует активность, но специализацию (assigned) почти не улучшает.

### 3.3 3-layer, first_spike_only=1 + homeostasis (target=7800; 7000–8200)
- run_id: 20260505T101258Z_c4c431028516
  - spikes/sample: 7259.04
  - assigned: 16/8192
  - neurons_per_class: mean=0.94, min=0, max=6, cv=1.945
  - вывод: активность стабилизирована, но обучение/специализация в глубине деградируют (слишком жёсткая разреженность в текущем виде).

### 3.4 3-layer, homeostasis с повышенным целевым коридором (target=30000; 25000–35000)
- run_id: 20260505T112546Z_f51c4ceb5f03
  - spikes/sample: 38349.97
  - assigned: 151/8192
  - neurons_per_class: mean=7.55, min=1, max=17, cv=0.622
  - вывод: рост активности улучшил равномерность распределения по классам (min=1), но assigned остаётся низким.

---

## 4) Изменения в коде (ключевые для статьи)

- Реализована поддержка 3-го сверточного слоя (Conv3) и greedy-обучения по слоям (Conv1→Conv2→Conv3).
- Исправлена логика ингибиции (EI), предотвращающая «обнуление» спайков и голодание глубоких слоёв.
- Добавлены:
  - activity-guard (ранний выход при «молчании» сети),
  - per-layer логирование активности (Conv1/Conv2/Conv3) в `activity.jsonl`,
  - метрика распределения нейронов по классам `neurons_per_class` в `label_map_summary`.
- Добавлен режим разреженности `first_spike_only` (один спайк на нейрон за предъявление) для диагностических абляций.
