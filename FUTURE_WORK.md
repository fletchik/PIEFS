# Future Work — нереализованные части из статьи

## 1. Graph Laplacian pretraining (статья Section 2.4)

**Что написано в статье:**
> "subsample a subset from training data and apply Graph-Laplacian approach to compute
> eigenfunctions on graph and to train the logistics regression with those features."
> "we utilise the value of cross-entropy loss of the logistics regression classifier
> as a target value T_class in equation 10"

**Зачем нужно:**
- Инициализирует φ_k вблизи правильного решения (избегает плохих локальных минимумов)
- Даёт осмысленное значение T_class для динамического взвешивания eq. 10
  (сейчас T_class=0.5 — эвристика, а должна быть CE логистической регрессии
  на Graph Laplacian признаках)

**Что нужно реализовать:**
1. Построить K-NN граф на обучающих данных
2. Вычислить собственные функции матрицы Лапласиана графа (scipy.sparse.linalg.eigsh)
3. Обучить логистическую регрессию на этих функциях → получить T_class
4. Инициализировать φ_k через дистилляцию от Graph Laplacian решений

**Файлы для создания:**
- `src/pretrain/graph_laplacian.py` — построение графа и вычисление eigenfunctions
- `src/pretrain/distill_init.py` — предобучение φ_k на GL-функциях
- Интеграция в `train.py` (опциональный этап до основного обучения)

---

## 2. Data augmentation для overfitting (статья Section 3, строки 261-268)

**Что написано:**
> "addition of points from wide normal distribution increases the degree of
> eigenfunction smoothness"
> "perturbation of input values can prevent overfitting"

**Зачем нужно:**
- BasisNet может зануляться на обучающих точках (piecewise-constant решение)
- Добавление случайных точек из широкого нормального распределения регуляризует
- Статья показывает это на примере единичной окружности (Fig. ??)

**Что нужно реализовать:**
- Augmentation wrapper для DataLoader: добавляет шум к x и/или дополнительные точки
- Флаг в конфиге `trainer.augment_noise_std` и `trainer.augment_extra_points`

---

## 3. Адаптивное T_orth на основе предобучения

**Текущее состояние:** T_orth=0.1 — эвристика.

**Идеальный вариант:** T_orth вычисляется из gram_error первых нескольких шагов
(warmup-измерение перед основным обучением), чтобы автоматически масштабироваться
под разные датасеты и размерности.
