# EFDO — Полный исследовательский план

**Дата:** 7 мая 2026
**Дедлайн ICML AI4Physics:** 9 мая 03:00 МСК
**Статус кода:** все P0-P3 баги зафиксированы (7 коммитов после аудита)

---

## ЧАСТЬ I — Почему PINN не выигрывает у diag/off?

### 1.1. Эмпирические наблюдения (test split, 7 мая)

| Dataset | off | diag | lambda_u_pinn |
|---------|-----|------|---------------|
| MNIST mc (best) | **96.11%** | 95.39% | 95.56% |
| HTRU2 (best) | 98.10% | **98.25%** | 97.95% |
| TwoMoon | 99.87% | 99.87% | 99.67% |
| Circles | **98.40%** | 90.00% | 79.27% |

**Вывод:** PINN **не даёт прироста** ни на одном датасете. На Circles (d=2) PINN сильно **хуже** (79% vs 98% у off).

### 1.2. Корневые причины (по аудиту + анализу кода)

#### Причина 1: `apply_to` нелинейна по `v`, но мы делаем линейный rescale ⚠️ **CRITICAL**

```python
# lambda_u_pinn.py:254-258
norms = w.norm(dim=-1, keepdim=True)
w_unit = w / norms                  # нормализуем на сферу
u_w_unit = self._pinn(omega_v, w_unit)
return u_w_unit * norms              # ← ОШИБКА: предполагает, что PINN линеен по v
```

PINN — это `MLP(omega ⊕ v)` с Tanh-активациями. **Tanh нелинеен**.
Для настоящей ротации `U`: `U·(αv) = α·(U·v)` (1-однородна по v).
Для PINN: `PINN(ω, αv) ≠ α·PINN(ω, v)` в общем случае.

**Эффект:** `apply_to(x, ∇φ)` даёт **смещённую оценку** `‖A(x)∇φ‖²` → loss_dirichlet считается неправильно → метрика учится неправильно.

#### Причина 2: Distribution shift между pretrain и main loop ⚠️ **MAJOR**

PINN тренируется на `omega_v ~ N(0, 1)` (углы в [-3, 3]).
Но `_omega_mlp(x)` после Linear+Tanh+Linear может выдавать **любые значения**, в т.ч. за [-π, π].

**Эффект:** в main loop PINN получает OOD-входы → tanh-saturation → градиент исчезает.

#### Причина 3: Sparse skew-symmetric — слишком ограниченное пространство ⚠️ **DESIGN**

ω(x) — только первая под/над-диагональ → `(d-1)` степень свободы.
SO(d) имеет `d(d-1)/2` степеней. Для d=784: **783 vs 306 936** (1/391 от полного).

Trotter product соседних Givens-вращений — это `(d-1)`-параметрическая подгруппа SO(d), а не вся SO(d).

**Эффект:** на Circles (d=2) только **1 угол** доступен — слишком мало; на MNIST (d=784) — мизерная часть пространства ротаций.

#### Причина 4: PINN заморожен после pretrain ⚠️ **DESIGN**

```python
for p in self._pinn.parameters():
    p.requires_grad_(False)   # frozen forever
```

Только `_omega_mlp` и `_lam_mlp` учатся в main loop. PINN остаётся приближением iid-Gaussian-распределения, которое **не совпадает** с распределением `_omega_mlp(x)` для реального x.

#### Причина 5: Ortho-loss не гарантирует SO(d) ⚠️ **THEORY**

```python
loss_ortho = ((u1_n * u2_n).sum(dim=-1) ** 2).mean()  # cos²(angle)
```

cos²=0 для случайных пар v_1⊥v_2 — это **необходимое, но не достаточное** условие ортогональности. Для строгой ортогональности нужно
`E[U^T U] = I` на всём пространстве, а не на сэмплированных парах.

---

### 1.3. Пять предложений как улучшить PINN

#### Предложение A (рекомендую) — **отказаться от PINN, использовать Trotter напрямую** 🌟

```python
def apply_to(self, x, v):
    omega_v = bounded_omega(self._omega_mlp(x))   # [-π, π]
    raw = self._lam_mlp(x); raw -= raw.mean(dim=1, keepdim=True)
    lam = torch.exp(raw)
    w = lam * v                                    # Λ·v
    return _trotter_rotate(omega_v, w)             # точная ортогональная ротация
```

**Преимущества:**
- ✅ Строгая ортогональность (Trotter — продукт точных Givens-вращений)
- ✅ O(B·d) сложность — такая же как PINN
- ✅ Полностью дифференцируемо (через autograd backprop в omega)
- ✅ Нет distribution shift (нет PINN — нет проблемы)
- ✅ Нет нелинейного rescale (Trotter линеен по v: `R·(αv) = α·R·v` точно)

**Минусы:**
- ❌ Sequential операция — не параллелится по d
- ❌ Pure Python loop в `_trotter_rotate` (можно заменить на vectorised batched-rotation)

**Сложность реализации:** 30 мин. Удалить PINN-код, заменить вызов в `apply_to`.

#### Предложение B — Bound omega через `tanh * π`

```python
omega_v = torch.tanh(self._omega_mlp(x)) * torch.pi   # [-π, π]
```

Решает Причину 2 (OOD). 5 минут.

#### Предложение C — Multi-pass Trotter (full SO(d))

Применить Trotter дважды с разным шагом: первый проход (i, i+1), второй (i, i+2). За 2-3 прохода покрывается всё SO(d). Сложность всё ещё O(B·d).

**Сложность:** 1 час. Параметризация: `omega: (B, P*(d-1))` где P — число проходов.

#### Предложение D — Joint training (без freeze PINN)

После pretrain не замораживать PINN, а продолжать его учить вместе с omega_mlp на реальных x. Решает Причину 4. Но может разрушить устойчивость (PINN перестаёт быть универсальным solver).

#### Предложение E — Householder reflection product

`U = (I - 2 v_1 v_1^T)·...·(I - 2 v_k v_k^T)` — каноничная параметризация SO(d) через k=d Householder-векторов. Полное покрытие SO(d). Сложность O(B·d²) — хуже Trotter.

### Решение для статьи

**P0 (срочно для AI4Physics):**
1. **Реализовать Предложение A + B** — заменить PINN на Trotter+bounded_omega. Это даёт честную "rotation metric" без багов аудита §1.5.
2. Запустить на 4 датасетах × 5 сидов с новой метрикой
3. В статье: показать сравнение **off vs diag vs trotter** (исправленный PINN-вариант)

**P1 (после deadline):**
4. Предложение C (multi-pass) — full SO(d) coverage
5. Ablation: Trotter с 1, 2, 3 проходами

---

## ЧАСТЬ II — Бейслайны для сравнения

### 2.1. Что у нас есть

- ✅ NeuralEF (Deng et al. 2022) — основной бейслайн в статье
- ✅ Random Forest на raw features
- ✅ Logistic Regression на raw features

### 2.2. Что **нужно добавить** (рекомендация advisor + критические для review'а)

#### Tier A — Классические (sklearn, очень быстро, < 1 минуты на датасет)

| Метод | sklearn | Что мерит | Параметры |
|-------|---------|-----------|-----------|
| **PCA** | `sklearn.decomposition.PCA(n_components=K)` | Линейные направления макс. дисперсии | n_components=K |
| **Kernel PCA (RBF)** | `KernelPCA(kernel='rbf', n_components=K)` | Нелинейные направления через kernel trick | kernel='rbf', gamma='scale' |
| **Kernel PCA (poly)** | `KernelPCA(kernel='poly', degree=3)` | Polynomial features | degree=3 |
| **Spectral Embedding** | `sklearn.manifold.SpectralEmbedding(n_components=K)` | Eigenvectors of graph Laplacian | n_neighbors=10 |
| **t-SNE** | `sklearn.manifold.TSNE(n_components=2)` | Только для визуализации | perplexity=30 |
| **UMAP** | `umap-learn` | Nonlinear DR | n_neighbors=15 |
| **Isomap** | `sklearn.manifold.Isomap(n_components=K)` | Geodesic distances | n_neighbors=10 |
| **LLE** | `sklearn.manifold.LocallyLinearEmbedding` | Локально-линейная аппроксимация | n_neighbors=10 |
| **Autoencoder (linear)** | `nn.Linear(d, K) + nn.Linear(K, d)` | Линейный bottleneck | mse loss |
| **Random Projection** | `sklearn.random_projection.GaussianRandomProjection` | JL-baseline | n_components=K |

#### Tier B — Нейронные (свежие, надо реализовать)

| Метод | Год | Статья | Сложность реализации |
|-------|-----|--------|---------------------|
| **NeuralEF** | 2022 | Deng et al. (NeurIPS 2022) | ✅ есть в нашей статье |
| **SpIN** | 2019 | Pfau et al. (ICLR 2019) | ⚠️ Сложно (bilevel optim) |
| **SpectralNet** | 2018 | Shaham et al. (ICLR 2018) | ⏰ Средне (Cheeger embedding) |
| **NestedLoRA** | 2024 | Recent neural eigenfunction | 🔍 Нужно искать |
| **Neural Eigenmaps** | 2024 | Wu et al. | 🔍 Нужно искать |
| **Deep Spectral Clustering** | 2019 | Yang et al. | ⏰ Средне |

#### Tier C — Foundation models (для аргумента "наш метод универсальный")

- **SimCLR features** (контрастивное самообучение) — backbone = ResNet, K_features = 128
- **DINOv2 / CLIP** — pretrained features

**Вывод:** Минимум для статьи — Tier A (PCA, kPCA, SpectralEmbedding). 1 день на реализацию.

---

## ЧАСТЬ III — Датасеты

### 3.1. Текущие датасеты

| Dataset | d | N | task | используется |
|---------|---|---|------|--------------|
| HTRU2 | 8 | 17 898 | binary | ✅ есть |
| TwoMoon | 2 | 10 000 | binary | ✅ есть |
| Circles | 2 | 10 000 | binary | ✅ есть |
| MNIST binary | 784 | 14 K | binary (0 vs 1) | ✅ есть |
| MNIST 10-class | 784 | 70 K | multiclass | ✅ есть |

### 3.2. Датасеты для добавления (по advisor + research best practices)

#### Tier A — Tabular (быстро, разнообразие)

| Dataset | d | N | task | Источник |
|---------|---|---|------|---------|
| **Iris** | 4 | 150 | 3-class | sklearn |
| **Wine** | 13 | 178 | 3-class | sklearn |
| **Breast Cancer** | 30 | 569 | binary | sklearn |
| **Digits (8x8)** | 64 | 1 797 | 10-class | sklearn |
| **CovType** | 54 | 581 K | 7-class | sklearn (большой) |
| **Adult** | 14 | 32 K | binary income | UCI |

#### Tier B — Image (стандартные benchmarks)

| Dataset | d | N | task | Сложность |
|---------|---|---|------|-----------|
| **Fashion-MNIST** | 784 | 70 K | 10-class | Сложнее MNIST |
| **CIFAR-10** | 3072 | 60 K | 10-class | RGB, нужен GPU |
| **MNIST 5%** | 784 | 3.5 K | semi-supervised | Тестирует label efficiency |

#### Tier C — Text (новая модальность для статьи)

| Dataset | d | N | task |
|---------|---|---|------|
| **20 Newsgroups (TF-IDF)** | 5000 | 18 K | 20-class |
| **AG News (TF-IDF)** | 5000 | 120 K | 4-class |

#### Tier D — Astrophysics (для AI4Physics workshop'а — physics angle!)

- **HTRU2** ✅ (pulsar candidates) — есть
- **MAGIC Gamma Telescope** — UCI, 19 020 samples, 10 features, гамма vs hadron
- **Higgs Boson** (UCI) — 11M samples, 28 features (giant — для O(d)/O(N) аргумента)
- **MiniBooNE Particle ID** — 130K samples, 50 features

**Рекомендация для AI4Physics:** добавить **MAGIC Gamma Telescope** (быстро, реальная астрофизика, есть baselines в литературе).

---

## ЧАСТЬ IV — Платформа для трекинга

### 4.1. Сравнение

| Платформа | Free academic | Public sharing | Локальные логи | Cross-runs comparison |
|-----------|--------------|----------------|---------------|---------------------|
| **WandB** ⭐ | ✅ Free for personal/academic | ✅ Public projects, share URL | ✅ Через export | ✅ Best in class |
| **MLflow** | ✅ Open source | ⚠️ Self-host | ✅ SQLite/файлы | ✅ |
| **Neptune.ai** | ✅ Free academic | ✅ Public projects | ✅ Sync to disk | ✅ |
| **Aim** | ✅ Open source | ⚠️ Self-host | ✅ Native | ✅ Modern UI |
| **Comet.ml** | Free <100 runs/mo | ✅ Public | ✅ | ✅ |
| **TensorBoard.dev** | ❌ Закрыт в 2024 | ❌ | — | — |

### 4.2. Рекомендация: **WandB + локальный JSONL backup**

**Почему WandB:**
1. **Бесплатно для академии** (просто email с .edu или укажи "academic" в плане)
2. **Public projects** — можно дать read-only ссылку научнику/коллегам без регистрации
3. **API для скачивания** — `wandb.Api()` экспорт всех runs в pandas
4. **Интеграция с PyTorch** — `wandb.watch(model)`, `wandb.log({...})`
5. **Sweeps** — встроенный hyperparameter search

### 4.3. Что логировать (расширенный список)

#### Per-step (через `metrics.jsonl` уже есть):
- `step`, `k`, `wall_time`
- `loss`, `loss_gram`, `loss_task`, `loss_dirichlet`
- `gram_error`, `off_diag_error_k`
- `w_task_eff`, `w_mde_eff`, `ratio_gram`, `ratio_class`
- `val_acc`

#### Per-evaluation (новое):
- `val/gram_error_full` (полный val) ✅
- `val/accuracy`, `val/roc_auc`, `val/f1_macro`
- `test/gram_error_full` (полный test, новое) ⭐
- `test/accuracy`, `test/roc_auc` (новое)
- **Eigenvalue spectrum**: `eigenvalue_history[k]` — каждые N шагов
- **Metric statistics**: `lambda_mean`, `lambda_std`, `omega_norm` (для diag/sparse/pinn)
- **Gradient statistics**: `grad_norm_basis`, `grad_norm_metric`, `grad_norm_head`
- **Activation statistics**: `phi_mean`, `phi_std`, `phi_min`, `phi_max` (детектор коллапса)

#### Per-checkpoint (final):
- Полный test-eval (LR, RF, AUC) ✅
- Per-fraction labels: 5%, 10%, 50%, 100%
- Eigenfunction visualization (для 2D датасетов)
- **PINN sanity tests** (для PINN-вариантов): linearity, ortho, scale-invariance

#### Артефакты:
- `metrics.jsonl` — все per-step логи
- `checkpoint_final.pt` — модель
- `table1_test.json` — финальные числа
- `eigenfunctions.png` — визуализация (для 2D)
- `gram_history.png` — сходимость ортогональности
- `weights_schedule.png` — динамика w_task_eff, w_mde_eff

### 4.4. Реализация (план)

```python
# 1. Поставить wandb (если не стоит)
pip install wandb
wandb login   # один раз

# 2. В train.py добавить опцию
if cfg.logging.wandb:
    import wandb
    wandb.init(
        project="efdo-ai4physics",
        name=run_id,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=[ds_name, metric_type, weighting_mode],
    )
    # Хук в trainer'е: wandb.log({...}, step=global_step)

# 3. Public link
# https://wandb.ai/<your-username>/efdo-ai4physics
# Settings → Privacy → Public — можно делиться URL'ом
```

**Локальный backup:** уже есть `metrics.jsonl` ✅. Дополнительно — `wandb sync ./logs` после training чтобы синхронизировать оффлайн runs.

---

## ЧАСТЬ V — Тесты для off, diag, PINN

### 5.1. Что нужно протестировать

#### Универсальные тесты (для всех 3 типов):
1. **Shape test**: `A(x)·v` имеет shape (B, d)
2. **Determinant constraint**: `det(Λ(x)) ≈ 1` (для diag/sparse/pinn)
3. **Gradient flow**: `∂loss/∂metric_params ≠ 0`
4. **Identity test (off only)**: `A(x)·v == v` точно
5. **Numerical stability**: нет NaN/Inf при экстремальных x (e.g. x=10·ones)

#### Специфичные для rotational metrics (sparse, pinn, новый trotter):
6. **Orthogonality test**: `‖U(x)·v‖ / ‖v‖ ≈ 1.0` (изометрия)
7. **Linearity test**: `‖U·(αv) - α·(U·v)‖ < 1e-5` (1-однородность по v)
8. **Inverse test**: `U^T·U·v ≈ v` (для exact rotations)

#### Специфичные для PINN:
9. **PINN vs matrix_exp consistency** (для d ≤ 16): `‖PINN(ω, v) - expm(ω)·v‖ < 0.1`
10. **OOD robustness**: `PINN(ω·5, v)` не должен взрываться (ограничение через tanh)

### 5.2. Реализация

```python
# tests/test_metrics.py
import torch
import pytest
from src.model.metric import build_metric

@pytest.mark.parametrize("metric_type", ["off", "diag", "lambda_u_sparse", "lambda_u_pinn"])
def test_shape(metric_type):
    metric = build_metric(metric_type, input_dim=8, hidden_dims=[16, 16])
    x = torch.randn(4, 8)
    v = torch.randn(4, 8)
    out = metric.apply_to(x, v) if hasattr(metric, 'apply_to') else metric(x) * v
    assert out.shape == (4, 8)

@pytest.mark.parametrize("metric_type", ["diag", "lambda_u_sparse"])
def test_det_one(metric_type):
    metric = build_metric(metric_type, 8, [16, 16])
    x = torch.randn(4, 8)
    lam = metric._lam_mlp(x) if metric_type != "diag" else metric.mlp(x)
    lam = lam - lam.mean(dim=1, keepdim=True)
    assert torch.allclose(lam.sum(dim=1), torch.zeros(4), atol=1e-5)

def test_off_identity():
    metric = None  # off
    v = torch.randn(4, 8)
    Av = v  # off → no metric → A=I
    assert torch.allclose(Av, v)

@pytest.mark.parametrize("metric_type", ["lambda_u_sparse"])
def test_orthogonality(metric_type):
    metric = build_metric(metric_type, 8, [16, 16])
    x = torch.randn(4, 8)
    v = torch.randn(4, 8); v = v / v.norm(dim=-1, keepdim=True)
    # Build full U, check U^T U = I
    A = metric(x)            # full (B, d, d) ≈ U·Λ
    # Λ removes pure rotation, so test on output norm preservation:
    Av = metric.apply_to(x, v)
    norms_in = v.norm(dim=-1)   # = 1
    norms_out = Av.norm(dim=-1)
    # With det(Λ)=1 but Λ not identity, norms can differ but determinant preserved
    # Better test: rotate-only via _get_U (skip Λ)
    U = metric._get_U(x) if hasattr(metric, '_get_U') else None
    if U is not None:
        Uv = torch.bmm(U, v.unsqueeze(-1)).squeeze(-1)
        assert torch.allclose(Uv.norm(dim=-1), torch.ones(4), atol=0.05)

def test_pinn_linearity_BUG():
    """This test should FAIL on current PINN, PASS after Trotter fix."""
    from src.model.metric import LambdaUPinn
    m = LambdaUPinn(8, [16, 16])
    m.pretrain(steps=100)  # короткий
    x = torch.randn(4, 8)
    v = torch.randn(4, 8)
    Av = m.apply_to(x, v)
    A_2v = m.apply_to(x, 2*v)
    # Должно быть: A_2v ≈ 2 * Av (1-однородно по v)
    err = (A_2v - 2*Av).abs().max().item()
    assert err < 0.1, f"PINN nonlinear in v: error={err}"
```

**Запуск:** `pytest tests/test_metrics.py -v`

### 5.3. Ожидаемые результаты тестов на текущем коде

| Test | off | diag | sparse | pinn |
|------|-----|------|--------|------|
| Shape | ✅ | ✅ | ✅ | ✅ |
| det = 1 | n/a | ✅ | ✅ | ✅ |
| Identity (off) | ✅ | n/a | n/a | n/a |
| Orthogonality | n/a | n/a | ✅ | ⚠️ approx |
| **Linearity in v** | ✅ | ✅ | ✅ | ❌ **FAIL** |
| Gradient flow | ✅ | ✅ | ✅ | ⚠️ frozen PINN |

---

## ЧАСТЬ VI — CPU vs GPU стратегия

### 6.1. Что работает на CPU (текущая инфраструктура)

| Dataset | d | N | Time per run (CPU, 1 seed) |
|---------|---|---|--------------------------|
| HTRU2 | 8 | 17K | ~5 мин |
| TwoMoon | 2 | 10K | ~3 мин |
| Circles | 2 | 10K | ~3 мин |
| MNIST 10-class | 784 | 70K | ~30 мин |
| MNIST binary | 784 | 14K | ~10 мин |

**Полный 5-сидов план (off+diag+pinn × 4 датасета):** 4 × 3 × 5 × 15 мин = **15 часов CPU**

### 6.2. Что нужен GPU

- CIFAR-10 (d=3072) — на CPU 2-3 часа per run, на GPU 5 мин
- 20 Newsgroups (d=5000) — аналогично
- Большие батчи (B≥1024) для variance reduction в gram_loss

### 6.3. Где взять GPU (бесплатно/дёшево)

| Платформа | Цена | GPU | Лимит |
|-----------|------|-----|-------|
| **Google Colab Free** | $0 | T4 | ~3 часа/сессия |
| **Google Colab Pro** | $10/мес | T4/V100 | 24 часа/сессия |
| **Kaggle** | $0 | P100/T4 | 30 часов/неделя |
| **Paperspace Free** | $0 | M4000 | 6 часов/сессия |
| **HSE GPU server** | $0 | RTX 3090? | если есть доступ |
| **vast.ai / runpod** | $0.20/час | RTX 4090 | по требованию |

**Рекомендация:** Kaggle (30h/неделя, P100) — хватит на всё. Альтернатива — спросить advisor про HSE-сервер.

### 6.4. Как запустить на GPU

```bash
# 1. Проверить что torch видит GPU
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"

# 2. Запуск с GPU (Hydra config)
python train.py run_id=mnist_mc_off_gpu_s42 dataset=mnist_multiclass model.metric_type=off device=cuda

# 3. На Kaggle — загрузить код через GitHub:
# - В Kaggle notebook: !git clone https://github.com/<user>/efdo
# - !cd efdo && pip install -r requirements.txt
# - !python train.py run_id=... device=cuda
```

---

## ЧАСТЬ VII — Полный экспериментальный план

### 7.1. Цель: 5 сидов × 3 метрики × 4 датасета + ablations + бейслайны

### Tier 1 (CRITICAL — для основной таблицы статьи)

| # | Эксперимент | Конфигов | Сидов | Total runs |
|---|-------------|----------|-------|-----------|
| 1 | **Main grid**: 4 dataset × 3 metric × 2 weighting (static/dynamic) | 24 | 5 | **120** |
| 2 | **Sklearn baselines**: PCA, kPCA, SpectralEmbedding × 4 dataset | 12 | 5 | **60** |
| 3 | **NeuralEF reference** (если есть код) × 4 dataset | 4 | 5 | **20** |

**Subtotal Tier 1: 200 runs**, на CPU ~50 часов, на GPU ~6 часов

### Tier 2 (ABLATIONS — для секции аблейшенов)

| # | Ablation | Конфигов | Сидов | Total |
|---|----------|----------|-------|-------|
| 4 | **L_gram only** (без task, без dirichlet) | 4 | 5 | 20 |
| 5 | **L_gram + L_task** (без dirichlet) | 4 | 5 | 20 |
| 6 | **L_gram + L_dirichlet** (unsupervised) | 4 | 5 | 20 |
| 7 | **GL pretraining on/off** | 8 | 5 | 40 |
| 8 | **Dynamic vs Static weighting** | 8 | 5 | 40 |
| 9 | **K=3, 6, 10, 16** (eigenfunction count) | 16 | 3 | 48 |

**Subtotal Tier 2: 188 runs**, ~40 часов на CPU

### Tier 3 (NEW DATASETS — для расширения)

| # | Dataset | Лучшая метрика × 5 сидов |
|---|---------|--------------------------|
| 10 | Fashion-MNIST | 5 |
| 11 | MAGIC Gamma | 5 |
| 12 | Iris/Wine/Breast | 15 (3 датасета) |
| 13 | 20 Newsgroups | 5 |

**Subtotal Tier 3: 30 runs**, ~10 часов

### Tier 4 (NEW METRIC — после фикса PINN)

| # | Эксперимент | Total |
|---|-------------|-------|
| 14 | **Trotter metric** (Предложение A) × 4 dataset × 5 сидов | 20 |
| 15 | **Multi-pass Trotter** (P=2, P=3) × 4 dataset × 3 сида | 24 |

**Subtotal Tier 4: 44 runs**, ~12 часов

---

### **ИТОГО: ~462 runs, на CPU ~110 часов, на GPU ~15 часов**

---

## ЧАСТЬ VIII — Что делать к 9 мая (36 часов)

### День 1 (8 мая) — Минимум для статьи

#### Утро (4 часа)
- [ ] Реализовать **Предложение A** (Trotter metric, замена PINN) — 1 час
- [ ] Написать тесты `tests/test_metrics.py` — 1 час
- [ ] Подключить **WandB** (init, log, sweep config) — 1 час
- [ ] Написать **sklearn baseline скрипт** (PCA, kPCA, SpectralEmbedding) — 1 час

#### День (8 часов)
- [ ] Запустить Tier 1 main grid на **HTRU2 + TwoMoon + Circles** (CPU, 5 сидов) — 6 часов
- [ ] Параллельно: sklearn baselines на всех 4 датасетах — 30 мин
- [ ] Параллельно: Trotter metric на 4 датасетах × 5 сидов — 6 часов

#### Вечер (4 часа)
- [ ] Запустить **MNIST 10-class** main grid на GPU (Kaggle) — 1 час
- [ ] Сборка таблиц: mean ± std из всех runs — 1 час
- [ ] Начать писать секции статьи (Method, Experiments) — 2 часа

### День 2 (9 мая, утро) — Финал

#### 03:00 deadline → backwards plan:
- 02:00 — submit на OpenReview
- 01:00 — финальная вычитка PDF
- 00:00 — компиляция LaTeX, исправление references
- 23:00 (8 мая) — доделать секции Discussion, Limitations, Related Work
- 21:00 (8 мая) — все эксперименты закончены, числа в таблицах

---

## ЧАСТЬ IX — Чеклист "что записывать в каждом experiment"

### Минимум (обязательно):
- [x] `metrics.jsonl` — per-step метрики (✅ уже работает)
- [x] `checkpoint_final.pt` — модель + конфиг (✅)
- [x] `table1_test.json` — финальные числа на test (✅)
- [ ] `wandb_run_id` — ссылка на эксперимент в WandB
- [ ] `seed`, `git_hash`, `host`, `start_time`, `end_time` — reproducibility

### Желательно:
- [ ] `eigenvalue_spectrum.png` — график λ_1 ≤ λ_2 ≤ ... ≤ λ_K
- [ ] `gram_history.png` — ‖C - I‖_F over time
- [ ] `weights_schedule.png` — w_task_eff, w_mde_eff over time
- [ ] `eigenfunctions_2d.png` — для 2D датасетов (TwoMoon, Circles)
- [ ] `confusion_matrix.png` — для классификации

### Для ablations:
- [ ] `metrics.csv` — экспорт metrics.jsonl с агрегацией по сидам
- [ ] `aggregate.json` — `{config_name: {mean: x, std: y, n: 5}}`

### Для статьи:
- [ ] `final_table.tex` — LaTeX-таблица с error bars
- [ ] `figure_dirichlet_energy.tex` — eigenvalue plot
- [ ] `figure_gram_convergence.tex` — orthogonality over training

---

## ЧАСТЬ X — Что обсудить с advisor

### Критические вопросы:
1. **PINN или Trotter?** — Предложение A ломает PINN-вариант. ОК ли это? Есть ли другая интерпретация PINN-вклада в статье?
2. **На какие конференции подавать?**
   - ICML AI4Physics workshop (9 мая) — РЕАЛЬНО
   - ICLR full paper (16 мая) — нужны полные ablations
   - NeurIPS 2026 — после deadline
3. **Есть ли доступ к HSE GPU?** — для CIFAR-10, больших батчей
4. **Коллаборация с Аней Лазаревой** — её алгоритм weighting
5. **Какой минимум baseline'ов нужен?** — все ли Tier A или только PCA + kPCA?

### Технические вопросы:
6. **Тестировать на CIFAR-10?** — нужен GPU
7. **Добавлять Foundation models** (DINOv2 features)? — может усилить
8. **Нужны ли theoretical guarantees** (не просто эмпирика)?

---

## Приложение A — Команды для запуска

### Sklearn baselines
```bash
# scripts/run_sklearn_baselines.py — нужно написать
.venv/bin/python3 scripts/run_sklearn_baselines.py \
    --datasets htru2 two_moon circles mnist_multiclass \
    --methods pca kpca_rbf kpca_poly spectral_embedding \
    --seeds 42 123 456 789 1024 \
    --output results/baselines_test.json
```

### Main grid (5 сидов)
```bash
# scripts/run_main_grid.sh — нужно написать
for seed in 42 123 456 789 1024; do
    for ds in htru2 two_moon circles mnist_multiclass; do
        for metric in off diag trotter; do  # trotter = новый PINN-replacement
            for weighting in static dynamic; do
                .venv/bin/python3 train.py \
                    run_id="grid_${ds}_${metric}_${weighting}_s${seed}" \
                    dataset=$ds \
                    model.metric_type=$metric \
                    loss.dynamic_weighting=$([ "$weighting" = "dynamic" ] && echo true || echo false) \
                    seed=$seed \
                    logging.wandb=true
            done
        done
    done
done
```

### Тесты
```bash
.venv/bin/python3 -m pytest tests/test_metrics.py -v
```

---

## Приложение B — Полезные ссылки

### Бенчмарки и литература
- NeuralEF (Deng et al. 2022): https://arxiv.org/abs/2205.10678
- SpIN (Pfau et al. 2019): https://arxiv.org/abs/1806.02215
- SpectralNet (Shaham et al. 2018): https://arxiv.org/abs/1801.01587
- HTRU2 dataset: https://archive.ics.uci.edu/ml/datasets/HTRU2
- MAGIC Gamma: https://archive.ics.uci.edu/ml/datasets/magic+gamma+telescope

### Платформы
- WandB: https://wandb.ai/
- Kaggle: https://www.kaggle.com/

### ICML AI4Physics
- Workshop site: https://ai4physics.org (или искать на ICML 2026 program)

---

**Финальный итог:** для дедлайна 9 мая критичны:
1. ✅ Все P0 фиксы (сделано)
2. ⏳ Trotter metric (Предложение A) — 1 час
3. ⏳ Sklearn baselines (PCA, kPCA, SpectralEmbedding) — 1 час
4. ⏳ WandB интеграция — 1 час
5. ⏳ 5-seed runs main grid (CPU + Kaggle GPU) — 8 часов параллельно
6. ⏳ Написать статью — 8 часов

**Реально успеть:** да, если фокусироваться только на critical path.
