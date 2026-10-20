import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from server.routes import router

load_dotenv()

app = FastAPI(
    title="CourtRoom AI API",
    description="Backend API and streaming simulation engine for CourtRoom AI",
    version="2.0.0",
)

# Enable CORS for local dev web app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Mount frontend production build if available
frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if frontend_dist.exists() and frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")


@app.get("/")
async def root():
    return {
        "name": "CourtRoom AI API",
        "version": "2.0.0",
        "docs": "/docs",
        "status": "ready",
    }
