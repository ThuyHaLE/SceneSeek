# app.py

"""
SceneSeek Backend

All routers already split into routers/*.py, each route returns 501 when not implemented.
Client (client.js) will catch 501 and fallback to mock data on the frontend.

To "activate" a route:
1. Open the corresponding router in routers/, implement the real logic.
2. Remove the line `raise NOT_IMPLEMENTED`.
3. Client will automatically detect HTTP 200 and use real data — no frontend changes needed.

Run:
    pip install fastapi uvicorn pydantic
    uvicorn app:app --reload --port 8000
"""

import os

os.environ["MALLOC_ARENA_MAX"] = "2"
os.environ["MALLOC_TRIM_THRESHOLD_"] = "65536"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import ctypes
import gc

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import state  # noqa: F401  import to trigger load dataset once at startup
import model_state  # noqa: F401  import to trigger load model + FAISS index once at startup

import torch
torch.set_num_threads(1)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="SceneSeek API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # narrow down to frontend domain when deploy production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware to release memory after each request, especially important for GPU memory management
import psutil
_process = psutil.Process(os.getpid())

STATIC_PREFIXES = ("/static/", "/assets/")

@app.middleware("http")
async def release_memory_middleware(request, call_next):
    response = await call_next(request)

    path = request.url.path
    if path.startswith(STATIC_PREFIXES):
        # Static files (images, JS, CSS) are lightweight and do not require garbage collection or malloc_trim, 
        # avoiding slowdowns in batch frame loading
        return response

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    rss = _process.memory_info().rss / 1e9
    print(f"[MEM] {request.url.path} -> RSS after cleanup: {rss:.2f} GB")
    return response

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
# Dataset (state.) and model/search index (model_state.) 
# already been loaded & cached 1 time when imported
# state / model_state above run - each router get their dependencies via deps.py,
# DO NOT use app.state anymore.

from routers.data_router import router as data_router
from routers.search_router import router as search_router
from routers.refine_router import router as refine_router
from routers.feedback_router import router as feedback_router
from routers.query_router import router as query_router
from routers.event_router import router as event_router

app.include_router(data_router)
app.include_router(search_router)
app.include_router(refine_router)
app.include_router(feedback_router)
app.include_router(query_router)
app.include_router(event_router)

# ---------------------------------------------------------------------------
# Static files — keyframe images
# ---------------------------------------------------------------------------

KEYFRAME_DIR = os.path.join(os.getcwd(), "static", "images", "key_frame_folder_reduced")
if os.path.isdir(KEYFRAME_DIR):
    app.mount(
        "/static/images/key_frame_folder_reduced",
        StaticFiles(directory=KEYFRAME_DIR),
        name="key_frame_folder_reduced",
    )

# ---------------------------------------------------------------------------
# Serve React production build
# (comment out when dev with 2 different server — Vite + FastAPI)
# Uncomment after running `npm run build` and wanting to deploy a single server.
# ---------------------------------------------------------------------------

dist_dir = os.path.join(os.path.dirname(__file__), "sceneseek-frontend", "dist")
if os.path.isdir(dist_dir):
    app.mount("/assets", StaticFiles(directory=os.path.join(dist_dir, "assets")), name="assets")

    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str):
        return FileResponse(os.path.join(dist_dir, "index.html"))

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)