# EFDO: Code, Theory, and Inconsistencies
## Full Reference Document for Paper Editing

---

## 1. Теоретическая база (что предлагает статья)

### 1.1 Постановка задачи

Дана выборка из распределения $p(\mathbf{x})$. Ищем $K$ функций
$\phi_1, \ldots, \phi_K : \mathbb{R}^d \to \mathbb{R}$, каждая параметризована нейронной сетью.

**Задача:** минимизировать модифицированную энергию Дирихле (MDE) последовательно по $k$:
$$\mathcal{E}_A[\phi_k] = \mathbb{E}_p\bigl[\|\mathbf{A}(\mathbf{x})\nabla\phi_k(\mathbf{x})\|^2\bigr]$$

при ограничении ортонормальности:
$$\mathbb{E}_p[\phi_\alpha(\mathbf{x})\,\phi_\beta(\mathbf{x})] = \delta_{\alpha\beta}$$

Когда $\mathbf{A} = \mathbf{I}$ — стандартная энергия Дирихле, соответствующая Лапласиану.

### 1.2 Три компонента потерь

**Gram-ортонормальность:**
$$\mathcal{L}_{\text{gram}} = \|C_k - I_k\|_F^2, \quad C_k = \mathbb{E}_p[\Phi_k \Phi_k^\top]$$
где $\Phi_k = [\phi_1, \ldots, \phi_k]^\top \in \mathbb{R}^k$. На практике: $C_k \approx \frac{1}{B}\Phi_k^\top \Phi_k$ по батчу, при этом $\phi_1,\ldots,\phi_{k-1}$ detached.

**Энергия Дирихле:**
$$\mathcal{L}_{\text{mde}} = \mathbb{E}_p\bigl[\|\mathbf{A}(\mathbf{x})\nabla\phi_k(\mathbf{x})\|^2\bigr]$$

**Классификация** (линейная голова):
$$\log p(l | \mathbf{x}) \propto b_l + \mathbf{w}_l^\top \Phi_K(\mathbf{x})$$

**Суммарная функция потерь:**
$$\mathcal{L} = w_{\text{gram}}\,\mathcal{L}_{\text{gram}} + \mathrm{sg}(w_{\text{class}})\,\mathcal{L}_{\text{class}} + \mathrm{sg}(w_{\text{mde}})\,\mathcal{L}_{\text{mde}}$$

`sg(·)` — stop-gradient, веса не участвуют в backprop.

### 1.3 Динамическое взвешивание

Иерархия обучения: ортонормальность → классификация → минимизация энергии.

$$\begin{cases}
w_{\text{gram}} &= w_{\text{gram}}^0 \\
w_{\text{class}} &= w_{\text{class}}^0 \cdot \exp(-\mathcal{L}_{\text{gram}} / T_{\text{orth}}) \\
w_{\text{mde}} &= w_{\text{mde}}^0 \cdot \exp\bigl(-\max(\mathcal{L}_{\text{gram}}/T_{\text{orth}},\ \mathcal{L}_{\text{class}}/T_{\text{class}})\bigr)
\end{cases}$$

$T_{\text{orth}}, T_{\text{class}}$ — температуры: когда потеря превышает температуру, соответствующий вес подавляется к нулю.

### 1.4 Три варианта метрики A(x)

| Вариант | Определение | Параметры |
|---------|-------------|-----------|
| `off` (identity) | $\mathbf{A} = \mathbf{I}$ | нет |
| `diag` | $\mathbf{A} = \text{diag}(\lambda_1(x),\ldots,\lambda_d(x))$ | MLP: $x \to \lambda(x)$, ограничение $\sum_i \log\lambda_i = 0$ |
| `lambda_u_trotter` | $\mathbf{A}(x) = \boldsymbol{\Lambda}(x)\,\mathbf{U}(x)$ (по статье) | MLP-λ + MLP-ω, U — произведение поворотов Гивенса |

**Trotter-конструкция U:**
$$\mathbf{U}(\mathbf{x}) = \prod_{(i,j)\in\mathcal{P}} G_{ij}(\theta_{ij}(\mathbf{x}))$$
где $G_{ij}(\theta)$ — поворот Гивенса в плоскости $(i, i+1)$. Произведение ортогональных матриц — ортогональная матрица ⟹ $\mathbf{U} \in SO(d)$ гарантировано без QR-разложения или Gram–Schmidt.

### 1.5 Вспомогательные компоненты

**Аугментация:**
- Gaussian noise: $\tilde{\mathbf{x}} = \mathbf{x} + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$
- Wide normal injection: $\tilde{p}(\mathbf{x}) = (1-\rho)\,p(\mathbf{x}) + \rho\,\mathcal{N}(0, \sigma_{\text{wide}}^2 I)$, $\sigma_{\text{wide}} \approx 3\hat\sigma_{\text{data}}$

**Graph Laplacian pretraining:** n_pre точек → kNN-граф → нормализованные лапласиановские eigenvectors → дистилляция в $\phi_1,\ldots,\phi_K$. Cross-entropy логистической регрессии на этих признаках используется как $T_{\text{class}}$.

---

## 2. Описание кода (что делает каждый модуль)

### 2.1 `src/model/basis/basis_set.py` — BasisSet

Хранит K независимых нейросетей-функций. В фазе $k$:
- `set_active(k)`: размораживает только $\phi_k$, все остальные заморожены
- `get_phi_matrix(x, k)`: возвращает `(phi_matrix, grad_phi_k)` где
  - `phi_matrix` ∈ (B, k): $\phi_1(x),\ldots,\phi_k(x)$, причём $\phi_1,\ldots,\phi_{k-1}$ detached
  - `grad_phi_k` ∈ (B, d): $\nabla_x \phi_k(x)$, вычислен через autograd

Архитектура каждой функции: MLP с `[64, 64, 64]` нейронами, ReLU, выход — скаляр.

### 2.2 `src/model/spectral_model.py` — SpectralModel

Forward pass:
```
x → basis_set.get_phi_matrix(x, k) → phi_matrix (B,k), grad_phi_k (B,d)
                                    ↓
phi_full (B, K): phi_matrix в первых k позициях, нули в k+1,...,K
                                    ↓
metric.apply_to(x, grad_phi_k) → Ag_pinn (B, d)     [если есть apply_to]
metric(x)                       → A (B, d) или (B, d, d) [иначе]
                                    ↓
head(phi_full, y) → loss_class, logits, probs
```

Диспетчеризация метрики: `hasattr(metric, 'apply_to')` → использует apply_to (Trotter, PINN); иначе — вызывает `metric(x)` напрямую (diag).

### 2.3 `src/model/metric/lambda_u_trotter.py` — LambdaUTrotter

**Параметры:**
- `_lam_mlp`: MLP $x \to \mathbb{R}^d$, затем mean-subtraction + exp → $\lambda(x)$, $\prod_i \lambda_i = 1$
- `_omega_mlp`: MLP $x \to \mathbb{R}^{(d-1) \cdot n\_passes}$ → углы поворотов, ограничены $\pi\tanh(\cdot) \in (-\pi, \pi)$

**apply_to(x, v) — центральный метод:**
```python
raw = _lam_mlp(x) - mean  →  lam = exp(raw)    # (B, d)
w = lam * v                                       # Λ·v, поэлементно
omega = _omega(x)                                 # (B, P, d-1)
return _trotter_rotate(omega, w)                  # U·(Λ·v)
```
Итого: возвращает $\mathbf{U}(\omega(x)) \cdot (\boldsymbol{\Lambda}(x) \cdot \mathbf{v})$, то есть $\mathbf{U}\boldsymbol{\Lambda}\mathbf{v}$.

**_trotter_rotate — векторизованная реализация:**
Два полусвипа вместо d итераций:
- Чётный: пары (0,1), (2,3), (4,5), ... — применяются параллельно
- Нечётный: пары (1,2), (3,4), (5,6), ... — применяются параллельно

Это снижает размер autograd-графа с O(d) до O(1) Python-итераций, давая ≈d/2× ускорение.

### 2.4 `src/loss/spectral_loss.py` — SpectralDirichletLoss

```python
# Gram loss
C_k = (1/B) * phi_matrix.T @ phi_matrix          # (k, k)
L_gram = ||C_k - I_k||_F²

# MDE loss — три случая:
if Ag_pinn is not None:
    L_mde = ||Ag_pinn||².mean()                   # Trotter/PINN: ||A∇φ||²
elif A is None:
    L_mde = ||grad_phi_k||².mean()                # off: ||∇φ||²
elif A.ndim == 2:
    L_mde = (A * grad_phi_k).pow(2).sum(-1).mean()  # diag: ||λ⊙∇φ||²
else:
    L_mde = ||A @ grad_phi_k||².mean()            # full matrix

# Dynamic weighting
w_gram_eff  = w_gram
w_task_eff  = w_task * exp(-L_gram / t_orth)
w_mde_eff   = w_mde  * exp(-max(L_gram/t_orth, L_task/t_class))
# stop-gradient на весах
total = w_gram_eff * L_gram + sg(w_task_eff) * L_task + sg(w_mde_eff) * L_mde
```

### 2.5 `src/trainer/sequential_trainer.py` — SequentialTrainer

Структура обучения:
```
for k = 1..K:
    basis_set.set_active(k)
    optimizer = Adam(trainable_params)
    for step in 0 .. total_steps//K:
        batch = next(train_loader) + augmentation
        forward → loss → backward → optimizer.step
        if step % log_step: evaluate(), log metrics
        if step % save_period: checkpoint
    freeze phi_k
    log eigenvalue
```

Ключевые детали:
- `total_steps // K` шагов на функцию (одинаково для всех k)
- Метрическая сеть (A(x)) остаётся trainable во всех фазах
- Голова классификации остаётся trainable во всех фазах
- При evaluation: Gram-error считается по ВСЕМУ валидационному набору, не батчам

### 2.6 `src/dataset/torchvision_flat.py` — TorchvisionFlatDataset

Для MNIST/Fashion-MNIST/CIFAR-10:
- Официальный `train` (60k) делится на train/val с `val_fraction=0.1`
- Официальный `test` (10k) используется как test
- Стандартизация: mean/std считается ТОЛЬКО по train_idx (не по всему train)
- Итог: train = 54k, val = 6k, test = 10k (для MNIST 70k)

---

## 3. Несостыковки: код vs статья vs внутренняя логика

---

### ❌ КРИТИЧЕСКАЯ: Порядок A(x) = ΛU vs UΛ делает U необучаемой

**Статья (§2.4):**
$$\mathbf{A}(\mathbf{x}) = \boldsymbol{\Lambda}(\mathbf{x})\,\mathbf{U}(\mathbf{x})$$
тогда $\mathbf{A}\mathbf{v} = \boldsymbol{\Lambda}(\mathbf{U}\mathbf{v})$ и $\|\mathbf{A}\mathbf{v}\|^2 = \|\boldsymbol{\Lambda}\mathbf{U}\mathbf{v}\|^2$ — зависит от U.

**Код (`apply_to` в LambdaUTrotter):**
```python
w = lam * v          # Λ·v
return trotter_rotate(omega, w)  # U·(Λ·v) = (UΛ)·v
```
Код возвращает $(UΛ)\mathbf{v}$, то есть $\mathbf{A}_{\text{код}} = \mathbf{U}\boldsymbol{\Lambda}$.

**Последствие:**
$$\|\mathbf{U}\boldsymbol{\Lambda}\mathbf{v}\|^2 = (\boldsymbol{\Lambda}\mathbf{v})^\top \mathbf{U}^\top \mathbf{U} (\boldsymbol{\Lambda}\mathbf{v}) = \|\boldsymbol{\Lambda}\mathbf{v}\|^2$$
поскольку $\mathbf{U}$ ортогональна. **Вращение не влияет на норму.** Это означает:
- $\mathcal{L}_{\text{mde}} = \|\mathbf{U}\boldsymbol{\Lambda}\nabla\phi\|^2 = \|\boldsymbol{\Lambda}\nabla\phi\|^2$ — **не зависит от параметров ω-MLP**
- Классификационная голова видит `phi_full` (выход basis_set), а не выход метрики — тоже не зависит от ω-MLP
- Gram-loss зависит только от basis_set — тоже не зависит от ω-MLP

**Вывод: параметры ω-MLP (углы поворотов U) никогда не обновляются.** Вариант `lambda_u_trotter` в коде математически эквивалентен `diag`.

**Правильно должно быть** (чтобы U обучалась):
$$\mathbf{A} = \boldsymbol{\Lambda}\mathbf{U} \Rightarrow \mathbf{A}\mathbf{v} = \boldsymbol{\Lambda}(\mathbf{U}\mathbf{v})$$
```python
u_v = trotter_rotate(omega, v)   # U·v  сначала
return lam * u_v                  # Λ·(U·v) = ΛUv
```
Тогда $\|\boldsymbol{\Lambda}\mathbf{U}\mathbf{v}\|^2$ действительно зависит от ω → ω-MLP получает градиент.

**Что делать в статье:** либо исправить код и переписать §2.4 с $A = UΛ$ и объяснением, что при этом $\|A\nabla\phi\|^2 = \|\Lambda\nabla\phi\|^2$ (тогда честно — trotter не даёт другой энергии, только обогащает представление), либо поменять порядок в коде на $A = ΛU$ (как в статье) и перезапустить trotter-эксперименты.

---

### ❌ ВАЖНАЯ: val_acc ≠ test_acc — все числа в статье могут быть validation

**Статья:** везде написано "test accuracy".

**Код:** `collect_grid_results.py` читает `val_acc` из `metrics.jsonl`. Trainer логирует `val_acc` — это accuracy на validation split (6k точек для MNIST), а не на официальном test split (10k точек для MNIST).

Разделение данных в коде:
- MNIST: train=54k, val=6k, test=10k — eval ведётся по val (6k)
- Официальный test (10k) не используется в trainer вообще

**Вывод:** цифры 94.6%, 86.73%, 87.11% — это валидационные, а не тестовые accuracy. Разница обычно 0.3–1%. Для публикации нужно либо переименовать "test" в "val" (и объяснить протокол), либо добавить отдельную оценку на test.

---

### ❌ ВАЖНАЯ: Противоречие в тексте: +9.6 pp vs +11.1 pp

**Абстракт:** "94.6% vs. 84.98%, a **+9.6 pp** gain" — арифметически верно: 94.6 − 84.98 = 9.62.

**Раздел Contributions (с. 3):** "outperforms NeuralEF on MNIST by **+11.1 pp**" — неверно.

**Что делать:** заменить `+11.1` на `+9.6` в Contributions.

---

### ❌ ВАЖНАЯ: Протокол разбивки данных не соответствует заявленному

**Статья (§3.1):** "All datasets are split **70% train / 10% val / 20% test**"

**Код для MNIST/FM:**
- Официальный train: 60k → train=54k (90%), val=6k (10%)
- Официальный test: 10k (используется как test)
- Итого: 54k / 6k / 10k = **77% / 8.6% / 14.3%** — не 70/10/20!

**Код для HTRU2/бинарных:** `train_fraction=0.7` — 70% train, но val вырезается дополнительно.

**Что делать:** привести точное описание разбивки: "For torchvision datasets we use the official train/test split (60k/10k for MNIST) and carve 10% of the training set for validation. For tabular datasets (HTRU2), we use 70/30 stratified split with 10% of train as val."

---

### ⚠️ СРЕДНЯЯ: "Follow the same protocol as NeuralEF" — некорректно

**Статья (§3.4):** "We follow the same protocol as NeuralEF: all 70,000 examples, 10-class softmax head, K=16 eigenfunctions, 60,000 gradient steps."

**Проблема:** NeuralEF — полностью **unsupervised** метод. Он не использует метки во время обучения, только при оценке (linear probe). EFDO использует метки через $\mathcal{L}_{\text{class}}$.

**Что делать:** переформулировать: "We use the same data (all 70,000 examples) and head (10-class softmax, K=16) as reported by NeuralEF, but EFDO is supervised — labels are used during training via $\mathcal{L}_{\text{class}}$, while NeuralEF is fully unsupervised."

---

### ⚠️ СРЕДНЯЯ: Таблица гиперпараметров — шаги для мультиклассовых задач

**Статья Appendix (Tab. hyperparams):** "Steps (multiclass): 120,000"

**Статья §3.1 (текст):** "Training runs for 60,000 gradient steps for binary datasets and **MNIST multiclass**, and **120,000 steps for Fashion-MNIST and CIFAR-10 features**."

**Противоречие:** таблица гиперпараметров говорит, что для всех multiclass задач 120k шагов, тогда как в тексте MNIST использует 60k.

**Что делать:** добавить строку в таблицу: "Steps (MNIST mc): 60,000; Steps (FM / CIFAR-10): 120,000".

---

### ⚠️ СРЕДНЯЯ: CIFAR-10 — устаревшие числа (2 seed → 3 seed)

**Статья §3.6:** "Preliminary results for the identity metric (2 seeds completed) yield **85.4% and 85.8%**"

**Реальный статус (на сейчас):** 3 seed завершены: 85.40%, 85.80%, 85.30% → mean = 85.5%

**Абстракт:** "≈85.6%" — вычислено как среднее первых двух: (85.4+85.8)/2 = 85.6. Сейчас правильно: (85.4+85.8+85.3)/3 = 85.5%.

**Что делать:** обновить после завершения всех 5 seed.

---

### ⚠️ СРЕДНЯЯ: "ResNet-style features" — неточно

**Статья:** "512-dimensional **ResNet-style** features extracted from CIFAR-10 by a pretrained convolutional backbone"

**Код (extract_cnn_features.py):** использует `torchvision.models.resnet18(pretrained=True)`, убирает последний classification layer.

**Что делать:** заменить "ResNet-style" на "ResNet-18 features" для воспроизводимости.

---

### ⚠️ СРЕДНЯЯ: Random Forest в коде использует 500 деревьев, в статье — 200

**Статья (§3.1):** "Random Forest (200 trees, scikit-learn default depth)"

**Код (run_sklearn_baselines.py):** `RandomForestClassifier(500, random_state=seed, n_jobs=-1)`

**Что делать:** привести в соответствие. RF(500) обычно лучше RF(200); нужно выбрать одно значение и указать его везде.

---

### ℹ️ МАЛАЯ: Zero-padding phi_full не описан в статье

**Статья:** "A linear head over $\phi_1,\ldots,\phi_K$ is trained with cross-entropy."

**Код:** В фазе k, голова видит `phi_full = [φ_1,...,φ_k, 0,...,0]` (K измерений, последние K−k = 0). Нули заменяются реальными функциями по мере обучения.

Это важная деталь реализации: голова обучается с нулями для ещё необученных функций. Это не ошибка, но могло бы быть упомянуто.

---

### ℹ️ МАЛАЯ: output_bias не упоминается в статье

**Код:** параметр `output_bias` у каждой basis function (BasisNet) — смещение в выходном слое. Старые чекпоинты имеют `bias=True`, новые эксперименты запускаются с `bias=False`.

Числа в статье (94.6% MNIST off seed=0) получены с `bias=True` (старый чекпоинт). Новые прогоны используют `bias=False`. Разница потенциально важна.

**Что делать:** зафиксировать `output_bias: false` для всех публикуемых результатов и упомянуть в Appendix.

---

### ℹ️ МАЛАЯ: Gram-матрица в статье — обозначение транспонирования

**Статья:** $C_k = \mathbb{E}_p[\Phi_k \Phi_k^\top]$, где $\Phi_k \in \mathbb{R}^k$ — вектор-столбец. Тогда $\Phi_k \Phi_k^\top \in \mathbb{R}^{k \times k}$ — внешнее произведение.

**Код:** `C_k = (1/B) * phi_matrix.T @ phi_matrix` где `phi_matrix` ∈ (B, k). Это $(B \times k)^\top (B \times k) = k \times k$ — правильно.

Обозначение совпадает. Но в тексте $\Phi_k \Phi_k^\top$ — это одна реализация, а не ожидание — следует писать $C_k = \mathbb{E}_p[\phi_k(x)\phi_k(x)^\top]$ или уточнять, что $\Phi_k$ — матрица по батчу.

---

### ℹ️ МАЛАЯ: Название "Trotter product" — терминология

**Статья:** называет конструкцию U "Trotter product" и ссылается на формулу $e^{A+B} \approx e^A e^B$.

**Реально в коде:** U — произведение Гивенса-поворотов, где углы задаются нейросетью. Это не Trotter в смысле приближения матричной экспоненты — это просто параметризация SO(d) через элементарные вращения.

Связь с "Trotter" есть только исторически (такой же тип разложения), но в коде нет `expm`. Название может вводить в заблуждение специалистов по численным методам.

**Что делать:** либо уточнить: "We use a Givens-rotation factorization of $\mathbf{U}$, inspired by Trotter-product approximations of matrix exponentials", либо просто написать "Givens-rotation parametrization."

---

## 4. Сводная таблица несостыковок

| # | Тип | Место | Описание | Приоритет |
|---|-----|-------|----------|-----------|
| 1 | Код↔Теория | `apply_to()` vs §2.4 | A=UΛ в коде vs A=ΛU в статье → U необучаема | 🔴 Критическая |
| 2 | Код↔Статья | trainer eval vs §3 | val_acc представлена как "test accuracy" | 🔴 Критическая |
| 3 | Статья↔Статья | Абстракт vs Contributions | +9.6 pp vs +11.1 pp | 🟠 Важная |
| 4 | Код↔Статья | TorchvisionFlat vs §3.1 | split 77/8.6/14.3% vs "70/10/20%" | 🟠 Важная |
| 5 | Статья | §3.4 | "same protocol as NeuralEF" — NeuralEF unsupervised | 🟠 Важная |
| 6 | Статья | Appendix Tab | "120k steps multiclass" — MNIST использует 60k | 🟠 Важная |
| 7 | Статья | §3.6, Abstract | "2 seeds / ≈85.6%" — сейчас 3 seed, mean=85.5% | 🟡 Средняя |
| 8 | Код↔Статья | extract_cnn_features.py | "ResNet-style" → "ResNet-18" | 🟡 Средняя |
| 9 | Код↔Статья | run_sklearn_baselines.py | RF(500 деревьев) vs RF(200) в тексте | 🟡 Средняя |
| 10 | Код | spectral_model.py | phi_full с нулями для future φ — не описано | ⚪ Малая |
| 11 | Код | train.py | output_bias=True/False меняет результаты — не описано | ⚪ Малая |
| 12 | Статья | §2.2 | Обозначение Gram-матрицы: одна реализация vs ожидание | ⚪ Малая |
| 13 | Статья | §2.4 | "Trotter product" vs Givens-rotation parametrization | ⚪ Малая |

---

## 5. Что сделать перед финальной версией статьи

### Обязательно (без этого нельзя публиковать)
1. **Уточнить что считается** — val или test accuracy. Добавить eval на official test.
2. **Исправить +11.1 pp → +9.6 pp** в Contributions.
3. **Исправить описание разбивки данных** (70/10/20 → реальные цифры).
4. **Объяснить или исправить порядок ΛU vs UΛ** — либо переписать §2.4, либо починить код и переобучить.

### Важно для честного сравнения
5. Переформулировать "same protocol as NeuralEF" (они unsupervised).
6. Исправить таблицу гиперпараметров: MNIST mc = 60k шагов, FM/CIFAR-10 = 120k.
7. Унифицировать RF: 200 или 500 деревьев, одно значение везде.
8. Заменить "ResNet-style" → "ResNet-18 (pretrained on ImageNet, penultimate layer features)".

### После завершения экспериментов
9. Обновить все числа с `*` (preliminary) на полные 5-seed results.
10. Зафиксировать `output_bias=False` для всех публикуемых прогонов.
11. Добавить sklearn baselines для FM и CIFAR-10 в таблицу.

---

## 6. Параметры экспериментов (актуальные)

| Параметр | Значение |
|----------|---------|
| Оптимизатор | Adam, lr=1e-3, β=(0.9, 0.999), wd=0 |
| Batch size | 256 |
| K (binary) | 6 |
| K (multiclass) | 16 |
| Шаги (binary, MNIST mc) | 60,000 total → 10,000/функцию (binary), 3,750/функцию (MNIST) |
| Шаги (FM, CIFAR-10) | 120,000 total → 7,500/функцию |
| BasisNet | [64,64,64] × ReLU, скалярный выход |
| MetricNet | [64,64] × ReLU |
| w_gram, w_class, w_mde | 1.0, 1.0, 1.0 |
| Dynamic weighting | T_orth=0.1, T_class=0.5 |
| output_bias | False (новые прогоны) |
| Gradient clipping | отключён (default None) |
| GL pretraining | отключён (graph_laplacian: false) |
| Аугментация | отключена (noise_std=0, wide_normal_fraction=0) |
| Устройство | Apple M-series CPU (auto), или GPU на Colab |

---

*Документ создан: 2026-05-08. Актуален для состояния кода на commit 4d0ca99.*
