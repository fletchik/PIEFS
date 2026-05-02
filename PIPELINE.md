# EFDO Pipeline — краткое руководство

## Что делает этот код

Обучение **последовательных собственных функций** оператора Дирихле с обучаемой метрикой A(x).

```
для k = 1, 2, ..., K:
    обучить φ_k под потерей L_gram + L_mde + L_task
    заморозить φ_k
```

Результат: K функций, ортонормированных по данным, с минимальной энергией Дирихле.

---

## Структура кода

```
train.py                       ← точка входа (Hydra)
src/
  configs/
    train.yaml                 ← главный конфиг (K, total_steps, device, seed...)
    dataset/                   ← датасеты: two_moon, circles, mnist_binary, ...
    criterion/spectral.yaml    ← веса потерь + dynamic_weighting
    optimizer/adam.yaml        ← lr, betas
  model/
    basis/basis_function.py    ← φ_k(x) — сеть + autograd-градиент ∇φ_k
    basis/basis_set.py         ← K функций, управление freeze/unfreeze
    spectral_model.py          ← SpectralModel + BinaryHead / MulticlassHead
    metric/
      diag_metric.py           ← DiagMetric: A(x) = Λ(x), det=1
      lambda_u_sparse.py       ← LambdaUSparse: A = Λ(x)·U(x), U = expm(ω)
      lambda_u_pinn.py         ← LambdaUPinn: U аппроксимируется PINN
      metric_net.py            ← build_metric('off'|'diag'|'lambda_u_sparse'|...)
  trainer/
    sequential_trainer.py      ← основной тренировочный цикл
  loss/spectral_loss.py        ← L_gram + L_mde + L_task (статическое или динамическое взвешивание)
  logger/experiment_logger.py  ← запись _config.md и _results.md
```

---

## Функция потерь

```
L_gram     = ||C_k − I||²_F        где  C_k = (1/B) Φ^T Φ  — матрица Грама
L_mde      = mean ||A(x)∇φ_k||²   MDE (Modified Dirichlet Energy, ст. eq.5-6)
L_task     = BCE или CE            кросс-энтропия классификатора

L_total    = w_gram · L_gram  +  w_task_eff · L_task  +  w_mde_eff · L_mde
```

### Динамическое взвешивание (статья eq. 9-10, `dynamic_weighting: true`)

```
w_gram_eff  = w_gram   (всегда 1)
w_task_eff  = w_task  · exp(−gram_error / t_orth)
w_mde_eff   = w_dirichlet · exp(−max(gram_error/t_orth,  L_task/t_class))
```

Смысл: вначале активна только `L_gram` (ортогонализация). Как только `gram_error < t_orth`
начинает включаться `L_task`. Когда и `L_task < t_class` — включается `L_mde`.
Это "warm-up stage" из статьи, критичный для избежания локальных минимумов.

### Матрица A(x)

| `metric_type` | Формула A(x) | Смысл |
|---|---|---|
| `off` | I (нет метрики) | стандартная энергия Дирихле `‖∇φ‖²` |
| `diag` | Λ(x) диагональная, det=1 | масштаб по осям, обучаемый |
| `lambda_u_sparse` | Λ(x)·U(x), U=expm(ω) | полное вращение + масштаб, U точный |
| `lambda_u_pinn` | Λ(x)·U(x), U≈PINN | то же, но U аппроксимируется нейросетью |

---

## Запуск обучения

```bash
# Базовый запуск
python train.py run_id=exp01

# С параметрами
python train.py run_id=exp01 \
  dataset=two_moon \
  model.K=6 \
  model.metric_type=diag \
  trainer.total_steps=60000 \
  trainer.seed=42 \
  criterion.dynamic_weighting=true \
  writer.mode=disabled

# С динамическим взвешиванием (рекомендуется, соответствует статье)
python train.py run_id=exp_dyn \
  criterion.dynamic_weighting=true \
  criterion.t_orth=0.1 \
  criterion.t_class=0.5

# Продолжение с чекпоинта
python train.py run_id=exp01 +resume=logs/exp01/checkpoint_30k.pt
```

### Параметры конфига

| Параметр | Дефолт | Описание |
|---|---|---|
| `model.K` | 6 | число собственных функций |
| `model.metric_type` | `'off'` | тип метрики A(x) |
| `trainer.total_steps` | 60000 | шагов всего (= K × шагов/функцию) |
| `trainer.seed` | 42 | seed для воспроизводимости |
| `criterion.dynamic_weighting` | false | eq.10 из статьи |
| `criterion.t_orth` | 0.1 | порог gram_error для включения L_task |
| `criterion.t_class` | 0.5 | порог L_task для включения L_mde |

---

## Оценка результатов

```bash
# Из чекпоинта
python scripts/eval_from_checkpoint.py logs/exp01/checkpoint_final.pt

# Ключевые метрики в чекпоинте:
#   gram_error_final   — ‖C−I‖_F по всему val-сету (ортогональность)
#   eigenvalue_history — энергии Дирихле φ_1,...,φ_K (прокси для λ_k)
#   val/accuracy       — точность классификатора на val
#   val/roc_auc        — ROC-AUC
```

---

## Группы экспериментов

| Группа | Датасет | Метрики | K | Цель |
|---|---|---|---|---|
| A | Two-moon, Circles | off, diag, sparse | 6 | базовая 2D валидация |
| B | MNIST binary (0 vs 1) | off, diag, sparse | 6, 16 | бинарная классификация |
| C | MNIST 10 классов (без меток) | off, diag, sparse | 10 | unsupervised features |
| D | MNIST 10 классов (с метками) | off, diag | 10 | supervised + MDE |
| E | HTRU2 | off, diag, sparse | 6 | реальные данные (статья Table 1) |
| F | CIFAR-10 binary | off, diag | 6, 16, 32 | высокая размерность |

Запуск всех: `bash scripts/reproduce_all.sh`
Запуск отдельной группы: `GROUPS="A" bash scripts/reproduce_all.sh`

---

## Чекпоинты и логи

```
logs/
  <run_id>/
    checkpoint_30k.pt       ← промежуточный
    checkpoint_final.pt     ← финальный (все K функций обучены)
    checkpoint_best_val.pt  ← лучший val_accuracy
  <run_id>_config.md        ← конфиг до запуска
  <run_id>_results.md       ← метрики после
```

Чекпоинт содержит: `model_state_dict`, `optimizer_state_dict`, `config`,
`eigenvalue_history`, `gram_error_history`, `wall_time_per_function`, `metrics_history`.

---

## Проверочные скрипты

```bash
python scripts/verify_pinn_rotation.py   # Check 1: геометрия LambdaUSparse/Pinn
python scripts/verify_gram.py            # Check 2: сходимость gram_error
python scripts/check3_smoke.py           # Check 3: smoke-тест всех голов
python scripts/verify_circle.py          # Check 4: корреляция с cos/sin на круге
```

Все 4 проходят ✅
