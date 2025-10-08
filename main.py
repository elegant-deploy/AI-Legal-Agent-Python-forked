from fastapi import FastAPI
from routes.legal_route import router as legal_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Legal Assistant API is running"}

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running and responsive.
    Returns system status, timestamp, and optional service info.
    """
    return {
        "status": "healthy",
        "service": "Legal Document Assistant API",
    }

app.include_router(legal_router, prefix="/legal", tags=["legal"])
