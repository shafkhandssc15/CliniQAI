"""Drop-in integration for CliniQ AI's FastAPI app.

1.  pip install -r requirements.txt   (xgboost, numpy, pandas, scikit-learn)
2.  python train.py                   (writes artifacts/)
3.  In app.py:   from integration_example import router as ml_router ; app.include_router(ml_router)

Adjust `adapt_for_consensus` to the exact dict shape your existing consensus core expects.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from cliniq_ml.engine import CliniQEngine

router = APIRouter(prefix="/api/ml", tags=["ml"])
MODEL_DIR = Path(__file__).parent / "artifacts"


@lru_cache(maxsize=1)
def get_engine() -> CliniQEngine:
    return CliniQEngine(MODEL_DIR)


class AnalyzeRequest(BaseModel):
    analytes: dict[str, float | str | None] = Field(..., description="OCR output, any common analyte names")
    age: float
    sex: str


@router.post("/analyze")
def analyze(req: AnalyzeRequest):
    try:
        return get_engine().analyze(req.analytes, age=req.age, sex=req.sex)
    except Exception as e:  # never let a model error become a silent 'healthy'
        raise HTTPException(status_code=500, detail=f"ML engine error: {e}")


def adapt_for_consensus(result: dict) -> dict:
    """Map the engine result onto a 'voter' record for the consensus core.

    Recommended: do NOT weight-average this with the VLMs. Treat POSITIVE/UNCERTAIN/NOT_ASSESSABLE as the
    calibrated, abstaining reference voter, and use VLM output only to (a) cross-check extracted values and
    (b) add context. Critical alerts must always escalate regardless of any vote.
    """
    ranked = sorted(((v["probability"], v["name"]) for v in result.get("conditions", {}).values() if v["decision"] == "POSITIVE"), reverse=True)
    return {
        "engine": "cliniq-ml",
        "model_version": result.get("model_version"),
        "status": result.get("status"),
        "primary_diagnosis": ranked[0][1] if ranked else None,
        "supported_conditions": [{"name": n, "probability": p} for p, n in ranked],
        "uncertain_conditions": result.get("uncertain", []),
        "not_assessable": result.get("not_assessable", []),
        "must_escalate": bool(result.get("critical_alerts")),
    }
