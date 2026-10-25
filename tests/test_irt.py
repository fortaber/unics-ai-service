import numpy as np

from app.irt import estimate_theta, calibrate_items, calibrate_2pl


def test_more_correct_gives_higher_theta():
    easy = [(1.0, 0.0, 1), (1.0, 0.0, 1), (1.0, 0.0, 1)]
    hard = [(1.0, 0.0, 0), (1.0, 0.0, 0), (1.0, 0.0, 0)]
    assert estimate_theta(easy)[0] > estimate_theta(hard)[0]


def test_se_shrinks_with_more_items():
    few_se = estimate_theta([(1.0, 0.0, 1)])[1]
    many_se = estimate_theta([(1.0, 0.0, 1)] * 10)[1]
    assert many_se < few_se


def test_no_responses_returns_prior_mean():
    theta, se = estimate_theta([], prior_theta=1.5, prior_se=0.5)
    assert abs(theta - 1.5) < 1e-6


def test_estimate_uses_discrimination():
    high = estimate_theta([(2.0, 0.0, 1)] * 3)[0]
    low = estimate_theta([(0.3, 0.0, 1)] * 3)[0]
    assert high > low


def test_calibrate_recovers_relative_difficulty():
    res = calibrate_items({10: [1, 1, 1, 1], 20: [0, 0, 0, 1]})
    assert res[10][0] < res[20][0]   # пройденный всеми пункт легче (меньше b)
    assert res[10][1] == 4           # n = число ответов


def _marginal_share(b):
    """Истинная доля верных по популяции N(0,1) - независимая от кода эталонная величина."""
    grid = np.linspace(-8, 8, 4001)
    w = np.exp(-0.5 * grid ** 2)
    w = w / w.sum()
    return float((w / (1.0 + np.exp(-(grid - b)))).sum())


def test_calibrate_is_unbiased_on_the_population_scale():
    """Раш-путь не должен СЖИМАТЬ трудность.

    Прежняя замкнутая формула b = ln((1-p)/p) верна для вероятности при theta = 0, а доля верных
    берется по популяции N(0,1). Из-за этого |b| сжималось примерно на шестую часть: при истинном
    2.0 выходило 1.69. Сжатия нет у 2PL-пути (MMLE), и задания с разным числом ответов ложились на
    разные шкалы, хотя CAT подбирает задание по близости b к theta через обе.

    Доля берется ТОЧНАЯ, без случайной выборки: проверяем систематический сдвиг, а не разброс.
    """
    for b_true in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0):
        share = _marginal_share(b_true)
        n = 1000
        correct = round(share * n)
        resp = [1] * correct + [0] * (n - correct)
        got = calibrate_items({1: resp})[1][0]
        assert abs(got - b_true) < 0.02, f"b={b_true}: получено {got}"


def test_calibrate_clips_degenerate_shares():
    """Все верно или все неверно - оценка упирается в границу, а не уходит в бесконечность."""
    assert calibrate_items({1: [1] * 20})[1][0] <= -2.0
    assert calibrate_items({2: [0] * 20})[2][0] >= 2.0


def _simulate_2pl(items, n_students, seed):
    """items = {item_ref: (a, b)}; популяция способностей N(0,1); вернуть rows (student, item, correct)."""
    rng = np.random.default_rng(seed)
    thetas = rng.normal(0.0, 1.0, n_students)
    rows = []
    for s in range(n_students):
        for ref, (a, b) in items.items():
            p = 1.0 / (1.0 + np.exp(-a * (thetas[s] - b)))
            rows.append((s, ref, 1 if rng.random() < p else 0))
    return rows


def test_calibrate_2pl_recovers_discrimination_order():
    # Дискриминация в MMLE опознаваема только при нескольких заданиях на студента (нужно разрешение
    # theta): к двум фокусным заданиям добавляем якорный банк (a=1.0 на разных b).
    bank = {10: (1.8, 0.0), 20: (0.4, 0.0),
            1: (1.0, -1.5), 2: (1.0, -0.5), 3: (1.0, 0.5),
            4: (1.0, 1.5), 5: (1.0, -1.0), 6: (1.0, 1.0)}
    rows = _simulate_2pl(bank, 400, seed=1)
    res = calibrate_2pl(rows)
    assert res[10][0] > res[20][0]   # выше истинная a -> выше оцененная a


def test_calibrate_2pl_recovers_difficulty_sign():
    rows = _simulate_2pl({10: (1.0, -1.5), 20: (1.0, 1.5)}, 400, seed=2)
    res = calibrate_2pl(rows)
    assert res[10][1] < res[20][1]   # легче -> ниже b


def test_calibrate_2pl_guard_thin_item():
    rows = _simulate_2pl({10: (1.5, 0.0)}, 5, seed=3)   # ниже MIN_RESPONSES_FOR_2PL
    res = calibrate_2pl(rows)
    assert res[10][0] == 1.0          # гард -> Rasch a=1
    assert res[10][2] == 5            # n = число ответов


def test_guard_refits_difficulty_by_mmle_not_by_share():
    """Ниже порога 2PL трудность пересчитывается MMLE, а не остается стартовой оценкой по доле.

    Раньше ветка гарда делала `continue`, и b навсегда оставалась оценкой по доле верных СВОЕГО
    задания. Комментарий и замер при этом утверждали, что там работает MMLE с a = 1 (найдено
    ревью). Из-за этого подъем порога переводил задания с 20..199 ответами не на MMLE, а на
    стартовую оценку.

    Признак различающий: у оценки по доле b зависит ТОЛЬКО от ответов своего задания. У MMLE она
    зависит и от остального банка, потому что банк двигает апостериор способности. Держим ответы
    фокусного задания неизменными и меняем окружение.
    """
    focus = [(s + 1, 1, 1 if s % 2 == 0 else 0) for s in range(40)]   # доля верных ровно 0.5

    easy_bank, hard_bank = [], []
    for s in range(40):
        for j, b in enumerate([-2.0, -1.8, -1.6], start=2):
            easy_bank.append((s + 1, j, 1 if s % 3 else 0))
        for j, b in enumerate([2.0, 1.8, 1.6], start=5):
            hard_bank.append((s + 1, j, 0 if s % 3 else 1))

    b_easy = calibrate_2pl(focus + easy_bank, min_n=200)[1][1]
    b_hard = calibrate_2pl(focus + hard_bank, min_n=200)[1][1]

    # Порог по факту: со стартовой оценкой обе величины выходят РОВНО одинаковыми (-0.0 и -0.0),
    # с MMLE расходятся на 0.025. Сдвиг мал, потому что банк небольшой, но он структурный:
    # у оценки по доле его не может быть в принципе.
    assert abs(b_easy - b_hard) > 0.01, (
        f'b фокусного задания не сдвинулась от смены банка ({b_easy} и {b_hard}) - '
        'значит ниже порога MMLE не работает, а отдается стартовая оценка по доле')


def test_guard_still_holds_at_a_hundred_responses():
    """Сто ответов - это ЕЩЕ гард Раша, а не 2PL.

    Порог поднят с 20 до 200 по замеру 2026-09-07: на данных с настоящей разной дискриминацией
    2PL начинает выигрывать у Раша только около двухсот ответов, а при 20-60 упирался в потолок
    дискриминации в 17-42 процентах прогонов. Тест сторожит именно это решение: вернуть порог к
    двадцати - значит снова включить модель там, где она хуже простой.
    """
    bank = {10: (1.8, 0.0), 20: (0.4, 0.0), 1: (1.0, -1.0), 2: (1.0, 1.0)}
    res = calibrate_2pl(_simulate_2pl(bank, 100, seed=4))
    assert all(v[0] == 1.0 for v in res.values()), 'при ста ответах включился 2PL'
    assert res[10][2] == 100
