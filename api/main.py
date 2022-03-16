import uvicorn

from fastapi import FastAPI
from starlette.responses import RedirectResponse


app = FastAPI(
    title="Serving Request Pizza Classifier",
    description="API to serve dynamically models and train them",
    version="1.0",
    openapi_url="/api/v1/openapi.json",
)


@app.get("/", include_in_schema=False)
async def home():
    """
    Home endpoint to redirect to docs.
    """
    return RedirectResponse("/docs")


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
