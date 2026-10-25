"""CAT-отбор: выбор следующего задания по максимуму информации Фишера 2PL при
текущей theta + стоп-логика. Чистые функции, без I/O. theta переоценивается тем же
EAP-оценщиком, что и в остальном сервисе (app.irt.estimate_theta)."""
from typing import List, Optional, Tuple

import numpy as np

from app.irt import estimate_theta, _2pl_p


def _fisher_information(theta: float, a: float, b: float) -> float:
    """Информация Фишера 2PL: I(theta) = a^2 * P * (1 - P)."""
    p = float(_2pl_p(np.array([theta]), a, b)[0])
    return (a ** 2) * p * (1.0 - p)


def next_item(responses: List[Tuple[float, float, int]],
              candidates: List[Tuple[int, float, float]],
              se_threshold: float, min_items: int, max_items: int,
              prior_theta: Optional[float] = None,
              prior_se: Optional[float] = None) -> dict:
    """responses = [(a,b,correct)] выданных; candidates = [(item_ref,a,b)] не выданных.
    Возвращает {theta, se, next_item_ref, stop, reason}."""
    n = len(responses)
    if n == 0:
        theta = 0.0 if prior_theta is None else float(prior_theta)
        se = 1.0 if not prior_se or prior_se <= 0 else float(prior_se)
    else:
        theta, se = estimate_theta(responses, prior_theta, prior_se)

    # Стоп-условия (min - нижняя граница; дальше SE / max / исчерпание банка).
    if n >= min_items and se < se_threshold:
        return {"theta": theta, "se": se, "next_item_ref": None,
                "stop": True, "reason": "se_reached"}
    if n >= max_items:
        return {"theta": theta, "se": se, "next_item_ref": None,
                "stop": True, "reason": "max_items"}
    if not candidates:
        return {"theta": theta, "se": se, "next_item_ref": None,
                "stop": True, "reason": "bank_exhausted"}

    # Отбор: максимум информации Фишера при текущей theta.
    best_ref, best_info = None, -1.0
    for item_ref, a, b in candidates:
        info = _fisher_information(theta, float(a), float(b))
        if info > best_info:
            best_info, best_ref = info, int(item_ref)
    return {"theta": theta, "se": se, "next_item_ref": best_ref,
            "stop": False, "reason": "continue"}
