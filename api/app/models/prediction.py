from pydantic import BaseModel


class PredResponse(BaseModel):
    """Response model for prediction."""
    prediction: str


class PredRequest(BaseModel):
    """Request model for prediction."""
    sample: str
