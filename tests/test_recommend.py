from app.recommend import rank


def _gap(eid, score=20, depth=0, theta_se=None, attempts_n=3):
    return {"element_id": eid, "depth": depth, "score": score, "band": 1,
            "theta": None, "theta_se": theta_se, "attempts_n": attempts_n}


def test_foundational_gap_ranks_above_deep_gap():
    res = rank([_gap(1, depth=5), _gap(2, depth=0)], 5)
    order = [r["element_id"] for r in res]
    assert order.index(2) < order.index(1)


def test_severe_gap_ranks_above_mild_gap():
    res = rank([_gap(1, score=80), _gap(2, score=10)], 5)
    order = [r["element_id"] for r in res]
    assert order.index(2) < order.index(1)


def test_confident_gap_ranks_above_uncertain():
    res = rank([_gap(1, theta_se=5.0), _gap(2, theta_se=0.1)], 5)
    order = [r["element_id"] for r in res]
    assert order.index(2) < order.index(1)


def test_mastered_yields_advancement():
    res = rank([{"element_id": 9, "depth": 1, "score": 95, "band": 3,
                 "theta": None, "theta_se": None, "attempts_n": 4}], 5)
    assert res[0]["kind"] == "advancement"
    assert res[0]["reason_code"] == "mastered_advance"


def test_remediation_outranks_advancement():
    skills = [{"element_id": 9, "depth": 1, "score": 95, "band": 3,
               "theta": None, "theta_se": None, "attempts_n": 4}, _gap(1, score=30)]
    res = rank(skills, 5)
    assert res[0]["element_id"] == 1
    assert res[0]["kind"] == "remediation"


def test_top_n_cap():
    res = rank([_gap(1), _gap(2), _gap(3), _gap(4)], 2)
    assert len(res) == 2


def test_reason_codes():
    res = rank([_gap(1, depth=0, score=30)], 5)   # depth 0 -> foundational
    assert res[0]["reason_code"] == "gap_foundational"
    res = rank([_gap(2, depth=4, score=10)], 5)   # deep + low score -> severe
    assert res[0]["reason_code"] == "gap_severe"
