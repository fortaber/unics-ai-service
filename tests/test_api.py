from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_estimate_contract():
    payload = {"model": "rasch", "responses": [{"difficulty": 0.0, "correct": 1}], "prior": None}
    r = client.post("/estimate", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "theta" in body and "se" in body


def test_calibrate_contract():
    payload = {"model": "rasch", "responses": [
        {"student_ref": 1, "item_ref": 10, "correct": 1},
        {"student_ref": 2, "item_ref": 10, "correct": 0},
    ]}
    r = client.post("/calibrate", json=payload)
    assert r.status_code == 200
    items = r.json()["items"]
    assert items[0]["item_ref"] == 10
    assert items[0]["n"] == 2


def test_auth_rejects_wrong_token(monkeypatch):
    monkeypatch.setenv("UNICS_IRT_TOKEN", "secret")
    r = client.post("/estimate", json={"model": "rasch", "responses": [], "prior": None})
    assert r.status_code == 401


def test_recommend_contract():
    payload = {"model": "content_v1", "top_n": 2, "skills": [
        {"element_id": 1, "depth": 0, "score": 20, "band": 1, "theta": None, "theta_se": None, "attempts_n": 3},
        {"element_id": 2, "depth": 4, "score": 80, "band": 1, "theta": None, "theta_se": None, "attempts_n": 3},
        {"element_id": 3, "depth": 1, "score": 95, "band": 3, "theta": None, "theta_se": None, "attempts_n": 4},
    ]}
    r = client.post("/recommend", json=payload)
    assert r.status_code == 200
    recs = r.json()["recommendations"]
    assert len(recs) == 2                       # top_n cap
    assert recs[0]["element_id"] == 1           # foundational severe gap first
    assert {"element_id", "kind", "priority", "reason_code"} <= set(recs[0].keys())


def test_recommend_auth_rejects_wrong_token(monkeypatch):
    monkeypatch.setenv("UNICS_IRT_TOKEN", "secret")
    r = client.post("/recommend", json={"model": "content_v1", "top_n": 3, "skills": []})
    assert r.status_code == 401


def test_estimate_accepts_discrimination():
    payload = {"model": "2pl",
               "responses": [{"discrimination": 1.5, "difficulty": 0.0, "correct": 1}],
               "prior": None}
    r = client.post("/estimate", json=payload)
    assert r.status_code == 200
    assert "theta" in r.json() and "se" in r.json()


def test_calibrate_2pl_returns_discrimination():
    """Дискриминация должна быть ОЦЕНЕНА, а не подставлена гардом.

    Фикстура давала 25 ответов - этого хватало при пороге 20, но после подъема порога до 200 оба
    задания попадали под гард Раша и возвращали ровно 1.0. Утверждение «поле есть» при этом
    оставалось верным, потому что у поля есть значение по умолчанию: тест молча перестал проверять
    то, ради чего написан (найдено ревью). Число ответов задается САМИМ порогом.
    """
    from app.irt import MIN_RESPONSES_FOR_2PL
    n = MIN_RESPONSES_FOR_2PL + 5
    responses = []
    for s in range(n):
        responses.append({"student_ref": s, "item_ref": 10, "correct": 1 if s % 2 == 0 else 0})
        responses.append({"student_ref": s, "item_ref": 20, "correct": 1 if s < n // 2 else 0})
    r = client.post("/calibrate", json={"model": "2pl", "responses": responses})
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 2
    assert all(it["n"] == n for it in items)
    assert any(abs(it["discrimination"] - 1.0) > 0.01 for it in items),         "дискриминация у обоих ровно 1.0 - значит сработал гард, а не оценка"


def test_cat_next_contract():
    payload = {
        "responses": [{"a": 1.0, "b": 0.0, "correct": 1}],
        "candidates": [{"item_ref": 101, "a": 1.0, "b": -2.0},
                       {"item_ref": 102, "a": 1.0, "b": 0.5}],
        "se_threshold": 0.0, "min_items": 10, "max_items": 20,
    }
    r = client.post("/cat/next", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert {"theta", "se", "next_item_ref", "stop", "reason"} <= set(body.keys())
    assert body["stop"] is False
    assert body["next_item_ref"] == 102


def test_cat_next_auth_rejects_wrong_token(monkeypatch):
    monkeypatch.setenv("UNICS_IRT_TOKEN", "secret")
    r = client.post("/cat/next", json={"responses": [], "candidates": [],
                                       "se_threshold": 0.0, "min_items": 5, "max_items": 20})
    assert r.status_code == 401
