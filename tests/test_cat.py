from app.cat import next_item


def test_selects_max_information_item():
    # theta после одного верного ответа ~ положительная; среди кандидатов максимум
    # информации Фишера при равном a даёт item с b ближе к theta.
    responses = [(1.0, 0.0, 1)]
    candidates = [(101, 1.0, -2.0), (102, 1.0, 0.5), (103, 1.0, 3.0)]
    res = next_item(responses, candidates, se_threshold=0.0, min_items=10, max_items=20)
    assert res["stop"] is False
    assert res["next_item_ref"] == 102  # b ближе всего к theta>0


def test_discrimination_raises_information():
    # При равном |theta-b| выше a -> выше информация -> выбирается.
    responses = [(1.0, 0.0, 1), (1.0, 0.0, 0)]  # theta ~ 0
    candidates = [(201, 0.4, 0.2), (202, 2.0, 0.2)]
    res = next_item(responses, candidates, se_threshold=0.0, min_items=10, max_items=20)
    assert res["next_item_ref"] == 202


def test_stops_on_se_threshold_after_min():
    # Достаточно согласованных смешанных ответов -> SE мало -> стоп по se_reached (после min).
    responses = [(1.5, 0.2, 1), (1.5, 0.2, 0)] * 4  # 8 ответов, SE < 0.5
    candidates = [(301, 1.0, 0.0)]
    res = next_item(responses, candidates, se_threshold=0.5, min_items=5, max_items=20)
    assert res["stop"] is True
    assert res["reason"] == "se_reached"
    assert res["next_item_ref"] is None


def test_does_not_stop_before_min_items():
    responses = [(1.5, 0.0, 1)] * 2
    candidates = [(401, 1.0, 0.0)]
    res = next_item(responses, candidates, se_threshold=10.0, min_items=5, max_items=20)
    assert res["stop"] is False


def test_stops_on_max_items():
    responses = [(1.0, 0.0, 1), (1.0, 0.0, 0)] * 3  # 6 ответов
    candidates = [(501, 1.0, 0.0)]
    res = next_item(responses, candidates, se_threshold=0.0, min_items=5, max_items=6)
    assert res["stop"] is True
    assert res["reason"] == "max_items"


def test_stops_on_bank_exhausted():
    responses = [(1.0, 0.0, 1)]
    res = next_item(responses, [], se_threshold=0.0, min_items=10, max_items=20)
    assert res["stop"] is True
    assert res["reason"] == "bank_exhausted"
    assert res["next_item_ref"] is None


def test_empty_responses_picks_central_item():
    # Старт сессии: theta = prior (0), выбираем item с b ближе к 0.
    candidates = [(601, 1.0, -3.0), (602, 1.0, 0.1), (603, 1.0, 2.5)]
    res = next_item([], candidates, se_threshold=0.0, min_items=5, max_items=20)
    assert res["stop"] is False
    assert res["next_item_ref"] == 602
