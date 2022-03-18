import uvicorn

from fastapi import FastAPI
from starlette.responses import RedirectResponse
from pathlib import Path

from app.models import PredRequest, PredResponse
from app.onnx_model import OnnxModel


app = FastAPI(
    title="Serving Request Pizza Classifier",
    description="API to serve dynamically models and train them",
    version="1.0",
    openapi_url="/api/v1/openapi.json",
)

path_to_model = Path("app/ml/model.onnx")
model = OnnxModel(path_to_model)


@app.get("/", include_in_schema=False)
async def home():
    """
    Home endpoint to redirect to docs.
    """
    return RedirectResponse("/docs")


@app.post("/predict", response_model=PredResponse, tags=["model"])
async def prediction(data: PredRequest):
    """
    Predict the label of the given text sample.

    Parameters
    ----------
    sample: PredRequest
        Text sample to predict.
    
    Returns
    -------
    PredResponse
        Prediction response.
    """
    sample = data.sample
    return {"prediction": model.predict(sample)}


@app.get("/healthz", status_code=200, include_in_schema=True, tags=["monitoring"])
async def healthz():
    """
    Healthz endpoint.
    """
    return {"status": "ok"}


@app.get("/readyz", status_code=200, include_in_schema=True, tags=["monitoring"])
async def readyz():
    """
    Readyz endpoint.
    """
    return {"status": "ready"}


if __name__ == "__main__":
    uvicorn.run("api.main:app", reload=True)
