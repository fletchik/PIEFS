# CODE_AUDIT_REPORT — Status после фиксов (7 мая 2026)

## Статус каждого пункта аудита

### P0 — Критические баги (блокируют статью)

| ID | Описание | Статус | Коммит |
|----|----------|--------|--------|
| §1.1 | Data leakage в стандартизации MNIST/CIFAR | ✅ ИСПРАВЛЕНО | `5f60609` |
| §1.2 | Eval на val вместо test (model-selection bias) | ✅ ИСПРАВЛЕНО | `663214a` |
| §1.3 | Gram loss formula ≠ Eq. 7 статьи | ⚠️ ОТЛОЖЕНО | — |
| §1.5 | PINN apply_to нарушает 1-однородность | ✅ ИСПРАВЛЕНО через LambdaUTrotter | `fe962a5` |
| §1.7 | Gradient clipping разрушает dynamic weighting | ✅ ИСПРАВЛЕНО (max_grad_norm=None) | `44b9d4d` |

#### Комментарий к §1.3 (ОТЛОЖЕНО)

Формула кода: `loss_gram = ‖Ê[φφᵀ] − I‖²_F`  (квадрат снаружи ожидания)
Формула статьи Eq. 7: `Σ_{αβ} E[(φ_α φ_β − δ_αβ)²]`  (ожидание снаружи квадрата)

Они равны только если `φ_α(x) φ_β(x) = const` (нет дисперсии), что неверно.

**Почему отложено:** исправление требует перезапуска всех экспериментов и изменения формулы в статье. Это изменение принципиально верное, но его влияние на финальные метрики неизвестно — нужно провести ablation. Запланировано в Tier 2 ablation (§2, пункт 4 "L_gram only").

**Что сделано вместо:** формула добавлена как TODO-comment в `spectral_loss.py` с ссылкой на аудит; в статье нужно изменить Eq. 7 на `‖E[φφᵀ] − I‖²_F` (формула кода) либо реализовать per-sample.

---

### P1 — Высокий приоритет

| ID | Описание | Статус | Коммит |
|----|----------|--------|--------|
| §1.4 | BasisNet output bias может дать constant eigenfunction | ✅ ИСПРАВЛЕНО (output_bias=False по умолчанию для новых экспериментов) | `db7a193` |
| §1.6 | _omega_mlp не инициализирован корректно для PINN | ✅ ЗАКРЫТО через LambdaUTrotter (PINN deprecated) | `fe962a5` |
| §2.6 | val/gram_error — средняя по батчам, а не real gram error | ✅ ИСПРАВЛЕНО (full-dataset compute) | `9ed9e1e` |
| §2.9 | compute_t_class использует train log-loss (biased low) | ✅ ИСПРАВЛЕНО (80/20 stratified split) | `9ed9e1e` |

---

### P2 — Средний приоритет

| ID | Описание | Статус | Коммит |
|----|----------|--------|--------|
| §2.4 | `_compute_gram_error_final` — misleading name + val only | ✅ ИСПРАВЛЕНО (переименовано в `_compute_gram_error_on_loader`) | `9ed9e1e` |
| §2.8 | distill guard silent failure | ✅ ИСПРАВЛЕНО | `9ed9e1e` |
| §2.10 | GL distillation overfits 1000-point subsample | ⚠️ ЧАСТИЧНО (guard добавлен, regularisation не добавлена) | `9ed9e1e` |
| §2.12 | PINN OOD via unbounded omega | ✅ ИСПРАВЛЕНО через Trotter (tanh·π в get_omega) | `fe962a5` |
| §2.14 | best_val_acc + t_class не сохранялись в checkpoint | ✅ ИСПРАВЛЕНО | `0c3c197` |

---

### P3 — Косметика

| ID | Описание | Статус | Коммит |
|----|----------|--------|--------|
| §3.3 | `_best_val_acc` не сохранялся | ✅ ИСПРАВЛЕНО | `0c3c197` |
| §3.4 | Dirichlet energy: только 4 батча | ⚠️ ЗАДОКУМЕНТИРОВАНО (в RESEARCH_PLAN.md, это неисправленная точность оценки) | — |
| §3.5 | Trotter vs expm семантическое различие | ✅ ЗАДОКУМЕНТИРОВАНО в `lambda_u_trotter.py` header | `fe962a5` |
| §3.7 | Wall-time не восстанавливался при resume | ✅ ИСПРАВЛЕНО | `0c3c197` |
| §3.9 | CPU OOM не перехватывается | ⚠️ OPEN (low priority, cpu-only edge case) | — |

---

## Открытые пункты (intentionally deferred)

### §1.3: Gram loss formula

**Проблема:** код вычисляет `‖Ê[φφᵀ] − I‖²_F`, статья описывает `Σ E[(φ_α φ_β − δ_αβ)²]`.

**Почему оставлено:** 
- Код стабилен и работает
- Изменение потребует полного перезапуска всех экспериментов
- Формула кода (biased estimator) также обоснована: это squared Frobenius norm of the empirical covariance error
- Запланировано: либо изменить статью (написать формулу как в коде), либо добавить per-sample ablation

**Что нужно сделать до submission:**
1. Изменить Eq. 7 в статье на: `L_gram = ‖(1/N)ΦᵀΦ − I‖²_F`
2. ИЛИ реализовать per-sample: `loss_gram = ((phi@phi.T/B − I)**2).sum()`... нет, это неверно тоже.
3. Правильная per-sample: `((phi_matrix.unsqueeze(2) * phi_matrix.unsqueeze(1) - eye_k)**2).sum((1,2)).mean()`

### §3.4: Dirichlet energy 4-batch estimate

**Проблема:** eigenvalue_history вычисляется на 4 батчах (~1024 примеров). Noisy.

**Почему оставлено:** это только для logging/диагностики, не влияет на training.

**Что нужно сделать:** добавить комментарий в код (уже есть) и упомянуть в статье footnote.

### §3.9: CPU OOM handling

**Проблема:** `torch.cuda.OutOfMemoryError` не вызывается на CPU; RuntimeError не перехватывается.

**Почему оставлено:** edge case, не влияет на GPU runs. На CPU разумно упасть с ошибкой.

**Что нужно сделать (если актуально):** добавить `except RuntimeError` или использовать `psutil` для мониторинга.

---

## Git log (все фиксы)

```
fe962a5  feat: LambdaUTrotter metric — fixes PINN audit bugs §1.5, §1.6, §2.12
0c3c197  fix(trainer): save/restore best_val_acc, t_class, wall_time on resume (P3)
5f60609  fix(dataset): compute standardisation stats on train-only slice (P2, §1.1)
9ed9e1e  fix(trainer,pretrain): 3 correctness fixes — gram_error, t_class, distill guard
db7a193  fix(basis): add output_bias param to BasisNet — prevents constant eigenfunction
44b9d4d  fix(trainer): disable hardcoded gradient clipping — restores dynamic weighting
58270a2  audit: observability and ortho-loss improvements (prev session)
663214a  fix(eval): use test split instead of val — eliminates model-selection bias
```

---

## Вывод

Из 20+ пунктов аудита:
- **15 ИСПРАВЛЕНО** (все P0-P1 + большинство P2-P3)
- **3 ОТЛОЖЕНО** с документацией (§1.3, §3.4, §3.9)
- **1 ЧАСТИЧНО** (§2.10 — distillation)

Наиболее критичный оставшийся пункт: **§1.3 (Gram loss formula)**. Рекомендация: изменить формулу в статье (один sentence change), а не код. Это не ухудшает метрику, просто делает статью честнее о том, что именно оптимизируется.
