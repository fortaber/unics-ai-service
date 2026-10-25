"""Content-based рекомендатель (content_v1): ранжирует навыки по тяжести пробела,
уверенности оценки и фундаментальности (глубине в дереве). Чистые функции, без I/O."""
from typing import Dict, List, Optional

# Полосы владения (совпадают с rolling_avg_estimator на стороне PHP).
BAND_GAP = 1
BAND_MASTERED = 3

# Веса content_v1.
W_SEVERITY = 0.5
W_CONFIDENCE = 0.2
W_FOUNDATIONAL = 0.3
W_ADV_SCORE = 0.3
BASE_REMEDIATION = 1.0
BASE_ADVANCEMENT = 0.0


def _confidence(theta_se: Optional[float], attempts_n: int) -> float:
    if theta_se is not None and float(theta_se) > 0:
        c = 1.0 / (1.0 + float(theta_se))
    else:
        c = min(int(attempts_n), 5) / 5.0
    return max(0.0, min(1.0, c))


def _foundational(depth: int) -> float:
    return 1.0 / (1.0 + max(0, int(depth)))


def rank(skills: List[Dict], top_n: int) -> List[Dict]:
    """skills: [{element_id, depth, score, band, theta|None, theta_se|None, attempts_n}].
    Возвращает <= top_n рекомендаций, отсортированных по priority desc."""
    out: List[Dict] = []
    for s in skills:
        band = int(s.get("band", 0))
        score = float(s.get("score", 0))
        conf = _confidence(s.get("theta_se"), int(s.get("attempts_n", 0)))
        if band == BAND_GAP:
            severity = max(0.0, min(1.0, (100.0 - score) / 100.0))
            found = _foundational(int(s.get("depth", 0)))
            priority = (BASE_REMEDIATION + W_SEVERITY * severity
                        + W_CONFIDENCE * conf + W_FOUNDATIONAL * found)
            if found >= 0.5:
                reason = "gap_foundational"
            elif severity >= 0.5:
                reason = "gap_severe"
            else:
                reason = "gap"
            out.append({"element_id": int(s["element_id"]), "kind": "remediation",
                        "priority": round(priority, 4), "reason_code": reason})
        elif band == BAND_MASTERED:
            priority = BASE_ADVANCEMENT + W_CONFIDENCE * conf + W_ADV_SCORE * (score / 100.0)
            out.append({"element_id": int(s["element_id"]), "kind": "advancement",
                        "priority": round(priority, 4), "reason_code": "mastered_advance"})
    out.sort(key=lambda r: r["priority"], reverse=True)
    return out[:max(0, int(top_n))]
