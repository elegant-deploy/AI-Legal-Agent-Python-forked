import sys
# Force line-buffered output so request logs (print) show in the uvicorn terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

from fastapi import FastAPI
from routes.legal_route import router as legal_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


@app.on_event("startup")
def warmup_chroma():
    """Warm Chroma client and embedding path so first request doesn't pay ~3s connection cost."""
    try:
        from agent.legal_agent import get_all_collections, embed_query
        names = get_all_collections()
        print(f"🔥 Chroma warmup: {len(names)} collection(s)")
        if names:
            embed_query("warmup")
            print("🔥 Embedding warmup done")
    except Exception as e:
        print(f"⚠️ Warmup skipped: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to Legal Assistant API v1.1.0"}

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running and responsive.
    Returns system status, timestamp, and optional service info.
    """
    return {
        "status": "healthy",
        "service": "Legal Document Assistant API",
        "version": "1.1.0"
    }

app.include_router(legal_router, prefix="/legal", tags=["legal"])
