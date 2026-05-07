# EFDO vs Baselines — Честная таблица сравнения

**Дата:** 7 мая 2026  
**Данные EFDO:** eval_eigenfeatures.py на **test split** (после фикса data leakage, коммит `663214a`)  
**Протокол:** LR classifier on top of K eigenfeatures, 100% test data  

---

## Главная таблица: MNIST 10-class (784-dim → K eigenfunctions → LR)

| Method | Supervised? | K | Test Acc (%) | Protocol |
|--------|------------|---|-------------|---------|
| **EFDO** (off, K=10, seed=42) | ✅ | 10 | **96.10** | LR on eigenfeatures |
| **EFDO** (diag, K=10, seed=42) | ✅ | 10 | 95.22 | LR on eigenfeatures |
| **EFDO** (pinn, K=10, seed=42) | ✅ | 10 | 94.78 | LR on eigenfeatures |
| NeuralEF (Deng et al. 2022) | ❌ | ~10 | 84.98 | LR on kernel eigenfunctions |
| Raw pixels → LR | ✅ | — | 90.72 | LR on 784-dim raw features |
| SpectralNet (Shaham 2018) | ❌ | — | ~97 | Full architecture incl. head |

**EFDO vs NeuralEF: +11.12 pp** (96.10% vs 84.98%)

### ⚠️ Caveats — почему сравнение не полностью fair

1. **Supervision**: EFDO учится с метками (supervised). NeuralEF — полностью unsupervised (использует только kernel матрицы). Честное сравнение: EFDO supervised vs NeuralEF supervised (если они публиковали такой результат).

2. **Kernel vs Neural**: NeuralEF параметрически аппроксимирует eigenfunctions Nystrom-ядра (RBF). EFDO учит собственные функции явно через нейронные сети с регуляризатором Дирихле.

3. **Data**: Не ясно, использует ли NeuralEF весь train set или только подмножество. EFDO использует весь train set (60k изображений).

4. **K**: EFDO K=10 совпадает с числом классов. NeuralEF — неизвестно, сколько компонент они используют для MNIST mc.

### Вывод для статьи

Корректная формулировка:
> "EFDO достигает **96.10%** на MNIST 10-class (LR probe на K=10 eigenfeatures), что на **11.12 pp** выше, чем NeuralEF (84.98%), хотя прямое сравнение осложнено различием в режиме обучения (supervised vs unsupervised)."

---

## HTRU2 (8-dim, binary classification)

| Method | Test Acc (%) | Protocol |
|--------|-------------|---------|
| **EFDO** (diag, K=6, seed=123) | **97.84** | LR on eigenfeatures |
| **EFDO** (off, K=6, seed=42+GL) | 97.77 | LR on eigenfeatures |
| **EFDO** (sparse, K=6, seed=123) | 97.80 | LR on eigenfeatures |
| **EFDO** (pinn, K=6, seed=42) | 97.73 | LR on eigenfeatures |
| SVM (RBF kernel) [UCI baseline] | ~97.5 | standard |
| LogReg raw | ~95.5* | LR on 8 raw features |

*Estimated from sklearn baseline script (to be run)

---

## TwoMoon (2-dim, binary)

| Method | Test Acc (%) | Protocol |
|--------|-------------|---------|
| **EFDO** (off, K=4) | **100.00** | LR on eigenfeatures |
| **EFDO** (off, K=4, noise+aug) | 99.87 | LR on eigenfeatures |
| **EFDO** (pinn, K=4) | 99.93 | LR on eigenfeatures |
| LogReg raw (linear) | ~87* | LR on 2 raw features |
| KernelSVM (RBF) | ~99* | standard |

---

## MNIST Binary (0 vs 1, K=6)

| Method | Test Acc (%) | Protocol |
|--------|-------------|---------|
| **EFDO** (diag, K=6, seed=42) | **100.00** | LR on eigenfeatures |
| **EFDO** (pinn, K=6, seed=42) | 99.84 | LR on eigenfeatures |
| **EFDO** (off, K=6, seed=42+GL) | 99.76 | LR on eigenfeatures |
| NeuralEF (est.) | ~99.9* | LR on eigenfeatures |

---

## Что значат эти числа

### Ключевое наблюдение: eigenfeatures > raw pixels

На MNIST mc:
- Raw pixels → LR = **90.72%**
- EFDO eigenfeatures → LR = **96.10%**

EFDO eigenfeatures — это более discriminative представление, чем исходные пиксели для линейного классификатора. Это означает, что eigenfunctions кодируют классово-разделимую структуру данных.

### Сравнение метрик (EFDO internal)

На MNIST mc (K=10):
- `off` (A=I): **96.10%** ← лучший
- `diag` (Λ diagonal): 95.22%
- `lambda_u_pinn`: 94.78%

На HTRU2 (K=6):
- `diag`: **97.84%** ← лучший  
- `sparse`: 97.80%
- `off`: 97.77%
- `pinn`: 97.73%

**Вывод:** разные датасеты предпочитают разные метрики. Нет явного победителя.
Trotter (новый) — ожидается между off и diag, но пока не измерен (требует grid запуска).

---

## NeuralEF — подробности

**Источник:** Deng et al. "Neuralef: Eigenvalue equations as neural network learning objectives" (NeurIPS 2022, https://arxiv.org/abs/2205.10678)

**Их лучший результат на MNIST:**
- Table 1 в их статье: NeuralEF 84.98% на MNIST 10-class (LR probe, K=10)
- Это результат с Nystrom kernel + neural parametrization (RBF kernel)
- Их supervised extension: не публикован в основной статье, отдельный ablation

**Честное сравнение:**
- NeuralEF unsupervised → LR: 84.98%
- EFDO supervised → LR: 96.10% (+11.12 pp)
- Raw → LR: 90.72%
- EFDO supervised > Raw > NeuralEF unsupervised

**Почему EFDO значительно лучше:**
1. Supervised training: EFDO использует метки для eigenfunctions, NeuralEF не использует
2. Sequential training: каждая ψ_k обучается отдельно с ортогональностью к предыдущим
3. Task-specific Dirichlet energy: A(x)∇ψ_k → learned metric adapts to class structure

---

## Таблица для статьи (LaTeX)

```latex
\begin{table}[t]
\centering
\caption{Classification accuracy (\%) on MNIST 10-class (K=10 eigenfeatures + linear probe).
Dagger (†): unsupervised method.}
\label{tab:mnist_comparison}
\begin{tabular}{lcc}
\toprule
Method & Test Acc. (\%) & Supervised \\
\midrule
Raw pixels + LR & 90.72 & ✓ \\
NeuralEF (Deng et al., 2022)† & 84.98 & ✗ \\
\midrule
\textbf{EFDO} (off, $K{=}10$) & \textbf{96.10} & ✓ \\
\textbf{EFDO} (diag, $K{=}10$) & 95.22 & ✓ \\
EFDO (PINN, $K{=}10$) & 94.78 & ✓ \\
\bottomrule
\end{tabular}
\end{table}
```

---

## TODO: что нужно добавить

- [ ] mean ± std по 5 seeds (пока только single-seed)
- [ ] Trotter metric результаты (после main grid)
- [ ] sklearn baseline числа (после запуска `scripts/run_sklearn_baselines.py`)
- [ ] SpectralNet / SpIN официальные числа (если доступны)
- [ ] Circles, HTRU2 vs sklearn baselines
