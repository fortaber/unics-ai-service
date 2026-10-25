from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ItemResponse(BaseModel):
    discrimination: float = 1.0
    difficulty: float
    correct: int = Field(ge=0, le=1)


class Prior(BaseModel):
    theta: float
    se: float


class EstimateRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model: str = "rasch"
    responses: List[ItemResponse]
    prior: Optional[Prior] = None


class EstimateResponse(BaseModel):
    theta: float
    se: float


class CalibrateItem(BaseModel):
    student_ref: int
    item_ref: int
    correct: int = Field(ge=0, le=1)


class CalibrateRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model: str = "rasch"
    responses: List[CalibrateItem]


class CalibratedItem(BaseModel):
    item_ref: int
    difficulty: float
    discrimination: float = 1.0
    n: int


class CalibrateResponse(BaseModel):
    items: List[CalibratedItem]


class SkillFeature(BaseModel):
    element_id: int
    depth: int = 0
    score: float = 0.0
    band: int = 0
    theta: Optional[float] = None
    theta_se: Optional[float] = None
    attempts_n: int = 0


class RecommendRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model: str = "content_v1"
    top_n: int = 3
    skills: List[SkillFeature]


class Recommendation(BaseModel):
    element_id: int
    kind: str
    priority: float
    reason_code: str


class RecommendResponse(BaseModel):
    recommendations: List[Recommendation]


class CatResponseItem(BaseModel):
    a: float = 1.0
    b: float
    correct: int = Field(ge=0, le=1)


class CatCandidate(BaseModel):
    item_ref: int
    a: float = 1.0
    b: float


class CatNextRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    responses: List[CatResponseItem]
    candidates: List[CatCandidate]
    se_threshold: float = 0.3
    min_items: int = 5
    max_items: int = 20
    prior: Optional[Prior] = None


class CatNextResponse(BaseModel):
    theta: float
    se: float
    next_item_ref: Optional[int] = None
    stop: bool
    reason: str
