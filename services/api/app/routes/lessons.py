from fastapi import APIRouter
from pydantic import BaseModel, Field
from ..semantic.lessons import promote_lesson

router = APIRouter(prefix="/lessons", tags=["lessons"])

class FeedbackIn(BaseModel):
    quality: str = Field(pattern="^(good|bad|unknown)$")
    reason: str | None = None

@router.post("/{point_id}/feedback")
def lessons_feedback(point_id: str, body: FeedbackIn):
    promote_lesson(point_id, body.quality)
    return {"ok": True}