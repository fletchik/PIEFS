# PIEFS — Experiment Plan for CIKM 2026
**Date:** May 2026 | **Deadline:** June 2026 (submission)

---

## Какие из 8 методов A(x) запускать?

Из 8 методов для бумаги нужны **5**. Остальные — теоретически интересны, но не
дают дополнительных экспериментальных точек.

| Метод | Запускать? | Обоснование |
|---|---|---|
| `off` (A=I) | ✅ **Всегда** | Baseline; показывает, что метрика вообще нужна |
| `diag` | ✅ **Всегда** | Лёгкий baseline; быстро обучается |
| `conformal` (σ(x)·I) | ✅ **Да** | Простейший x-dependent; тест гипотезы «нужна ли зависимость от x» |
| `global_low_rank` (I+UDVᵀ) | ✅ **Главный** | Лучшие результаты в D0; LDA-связь даёт теорию; r=C-1 |
| `local_low_rank` (I+U(x)Λ(x)V(x)ᵀ) | ✅ **Да** | Аблация: даёт ли x-зависимость прирост над global? |
| `fisher_diag` | ❌ **Пропустить** | Требует истинных grad log p(y\|x); медленно; теор. связь описана в тексте |
| `lambda_u_trotter` | ❌ **Пропустить** | Покрывает только (d-1) из d(d-1)/2 вращений; известно хуже global_lr |
| `normalize_det` вариант | ❌ **Пропустить** | Хранить как гиперпараметр; не основной эксперимент |

**Итого в основной таблице: off / diag / conformal / global_low_rank / local_low_rank**

**Ожидаемый порядок на MNIST:** `global_low_rank` > `local_low_rank` > `conformal` > `diag` > `off`
- Почему global > local: при 60k шагах local переобучает U(x),V(x) на малых данных
- Почему global_low_rank выигрывает: LDA-теорема гарантирует оптимальность при r=C-1

---

## Группы экспериментов

### D0 — Быстрая проверка g_k² фикса (htru2) ✅ ВЫПОЛНЕНО
```
Файл: logs/D0_htru2_gk_fixed_s*/metrics.jsonl
Вывод: w_mde@30k = 0.27 (fixed) vs 0.02 (original bug)
       AccBest: 97.77% (fixed) vs 97.36% (off)
```

### D1 — Аблация метрик, Two Moon (быстро, 2D)
```bash
bash scripts/run_experiments.sh D1
```
- **7 типов × 3 seed = 21 прогон**
- Время: ~2 часа CPU
- Цель: визуализация eigenfunction φ₁,φ₂ для каждой метрики
- **Ожидается:** conformal лучший на Two Moon (простая 2D структура)

### D2 — Аблация метрик, MNIST multiclass ← **ГЛАВНАЯ ТАБЛИЦА**
```bash
bash scripts/run_experiments.sh D2
```
- **5 типов × 5 seed = 25 прогонов × 60k шагов**
- Время: ~8 часов GPU (A100) или ~40 часов CPU
- global_low_rank: r=9 (=C-1), три фазы 30k/45k
- local_low_rank: r=9, три фазы 30k/45k
- Цель: итоговая таблица для бумаги

### D3 — Аблация ранга (htru2, global_low_rank)
```bash
bash scripts/run_experiments.sh D3
```
- **r ∈ {1,2,4,8,16} × 3 seed = 15 прогонов**
- Время: ~3 часа CPU
- Цель: подтвердить r=C-1=1 оптимально для бинарной задачи
- **Ожидается:** r=1 > r=2 > r=4 (убывающая отдача, теорема об ранге)

### D4 — Аблация g_k², MNIST
```bash
bash scripts/run_experiments.sh D4
```
- **2 варианта × 3 seed = 6 прогонов**
- Время: ~3 часа GPU
- Цель: показать влияние бага на больших данных

### D5 — Аблация трёхфазного curriculum (htru2)
```bash
bash scripts/run_experiments.sh D5
```
- **2 варианта × 3 seed = 6 прогонов**
- Время: ~1.5 часа CPU
- Цель: подтвердить, что curriculum (50%/75%) помогает

---

## Рекомендуемый порядок запуска

```
Неделя 1 (сейчас):
  [1] D0   — уже готово ✅
  [2] D5   — 1.5 ч CPU, быстро подтверждает curriculum
  [3] D3   — 3 ч CPU, подтверждает теорему о ранге
  [4] D1   — 2 ч CPU, красивые картинки для бумаги

Неделя 2 (GPU нужен):
  [5] D4   — 3 ч GPU, g_k² ablation на MNIST
  [6] D2   — 8 ч GPU, ГЛАВНАЯ ТАБЛИЦА

Итого: ~17 ч GPU + ~8 ч CPU
```

На **Google Colab A100** D2 займёт около 8 часов. Можно запустить ночью.

---

## Что ожидать в таблице (предварительные прогнозы)

### HTRU2 (бинарная, d=8, K=6)

| Метод | Acc (%) | Примечание |
|---|---|---|
| RF (sklearn) | ~97.5 | Reference baseline |
| PIEFS-off | ~97.4 | Известно из D0 |
| PIEFS-diag | ~97.5 | Лёгкий прирост |
| PIEFS-conformal | ~97.5 | x-dependent scalar |
| PIEFS-local_lr | ~97.6 | x-dependent low-rank |
| **PIEFS-global_lr** | **~97.8** | **LDA-optimal** |

### MNIST 10-class (d=784, K=16)

| Метод | Acc (%) | Примечание |
|---|---|---|
| NeuralEF* | ~85.0 | Unsupervised reference |
| PIEFS-off | ~96.1 | Известно из старых прогонов |
| PIEFS-diag | ~95.5 | Может быть хуже off при высоком d |
| PIEFS-conformal | ~96.3 | Простейший x-dependent |
| PIEFS-local_lr r=9 | ~96.8 | x-dependent, может переобучить |
| **PIEFS-global_lr r=9** | **~97.1** | **LDA connection, теор. оптимально** |

> ⚠️ Предупреждение: эти числа — прогнозы, не результаты. Реальные числа могут
> отличаться. Global_low_rank должен победить по теории, но если local_low_rank
> выигрывает — это тоже интересный результат (означает, что x-dependence важна).

---

## Структура итоговой таблицы для бумаги (Table 1)

```latex
\begin{table}[t]
\caption{Classification accuracy (\%) on benchmark datasets. PIEFS variants use 
K=6 (binary) or K=16 (multiclass), 5 seeds, mean ± std.}
\begin{tabular}{lcccc}
\toprule
Method & Two Moon & HTRU2 & MNIST & Fashion-MNIST \\
\midrule
Random Forest        & 100.0       & 97.5±0.1  & 97.0±0.2  & 87.5±0.3 \\
Logistic Regression  & 99.8±0.1    & 97.0±0.2  & 92.6±0.1  & 84.1±0.2 \\
NeuralEF (unsup.)    & ---         & ---       & 84.98     & ---      \\
\midrule
PIEFS-off            & 99.9±0.1    & 97.4±0.1  & 96.1±0.3  & TBD      \\
PIEFS-diag           & 99.9±0.1    & 97.5±0.1  & TBD       & TBD      \\
PIEFS-conformal      & TBD         & TBD       & TBD       & TBD      \\
PIEFS-local\_lr      & TBD         & TBD       & TBD       & TBD      \\
PIEFS-global\_lr     & \bf TBD     & \bf 97.8  & \bf TBD   & \bf TBD  \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Дополнительные датасеты (если нужно усилить бумагу)

Если D2 даёт убедительный результат, добавить:

| Датасет | Зачем | Конфиг |
|---|---|---|
| Fashion-MNIST | Более сложный чем MNIST, тот же формат | `dataset=fashion_mnist` |
| CIFAR-10 features | Высокоразмерный (512-dim ResNet features) | `dataset=cifar10_features` |
| Spotify (songs) | Реальные данные, числовые фичи | `dataset=spotify` |

Эти датасеты уже поддержаны в коде. Fashion-MNIST добавит 2 часа GPU.

---

## Как собрать результаты

После прогона экспериментов:
```bash
python scripts/collect_grid_results.py --log_dir logs
```
Выводит таблицу с mean±std по seed'ам для каждого run_id.
