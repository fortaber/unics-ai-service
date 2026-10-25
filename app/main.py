import os
from typing import Optional

from fastapi import FastAPI, Header, HTTPException

from app.irt import MIN_RESPONSES_FOR_2PL, calibrate_2pl, calibrate_items, estimate_theta
from app.recommend import rank
from app.cat import next_item
from app.models import (CalibratedItem, CalibrateRequest, CalibrateResponse,
                        EstimateRequest, EstimateResponse,
                        RecommendRequest, RecommendResponse, Recommendation,
                        CatNextRequest, CatNextResponse)

app = FastAPI(title="UNICS IRT service", version="1.0")


def _check_token(token: Optional[str]) -> None:
    expected = os.environ.get("UNICS_IRT_TOKEN")
    if expected and token != expected:
        raise HTTPException(status_code=401, detail="bad token")


@app.get("/health")
def health():
    """Здоровье плюс ПОРОГИ.

    Пороги отдаются наружу не для красоты: плагин держит их копию (item_irt_manager::
    MIN_N_FOR_2PL), потому что методисту надо показывать расстояние до 2PL. Копия держалась на
    одной лишь фразе «при изменении порога в сервисе править и здесь» - и разъехалась при первом
    же изменении. Теперь проверка здоровья сверяет их и говорит о расхождении вслух.
    """
    return {"status": "ok", "thresholds": {"min_responses_for_2pl": MIN_RESPONSES_FOR_2PL}}


@app.post("/estimate", response_model=EstimateResponse)
def estimate(req: EstimateRequest, x_unics_token: Optional[str] = Header(default=None)):
    _check_token(x_unics_token)
    responses = [(r.discrimination, r.difficulty, r.correct) for r in req.responses]
    prior_theta = req.prior.theta if req.prior else None
    prior_se = req.prior.se if req.prior else None
    theta, se = estimate_theta(responses, prior_theta, prior_se)
    return EstimateResponse(theta=theta, se=se)


@app.post("/calibrate", response_model=CalibrateResponse)
def calibrate(req: CalibrateRequest, x_unics_token: Optional[str] = Header(default=None)):
    _check_token(x_unics_token)
    if req.model == "2pl":
        rows = [(r.student_ref, r.item_ref, r.correct) for r in req.responses]
        fitted = calibrate_2pl(rows)
        items = [CalibratedItem(item_ref=ref, difficulty=b, discrimination=a, n=n)
                 for ref, (a, b, n) in fitted.items()]
    else:
        by_item: dict = {}
        for r in req.responses:
            by_item.setdefault(r.item_ref, []).append(r.correct)
        fitted = calibrate_items(by_item)
        items = [CalibratedItem(item_ref=ref, difficulty=b, n=n) for ref, (b, n) in fitted.items()]
    return CalibrateResponse(items=items)


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest, x_unics_token: Optional[str] = Header(default=None)):
    _check_token(x_unics_token)
    skills = [s.model_dump() for s in req.skills]
    recs = rank(skills, req.top_n)
    return RecommendResponse(recommendations=[Recommendation(**r) for r in recs])


@app.post("/cat/next", response_model=CatNextResponse)
def cat_next(req: CatNextRequest, x_unics_token: Optional[str] = Header(default=None)):
    _check_token(x_unics_token)
    responses = [(r.a, r.b, r.correct) for r in req.responses]
    candidates = [(c.item_ref, c.a, c.b) for c in req.candidates]
    prior_theta = req.prior.theta if req.prior else None
    prior_se = req.prior.se if req.prior else None
    res = next_item(responses, candidates, req.se_threshold, req.min_items,
                    req.max_items, prior_theta, prior_se)
    return CatNextResponse(**res)
