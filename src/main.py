"""
Main application entry point for DNA-based crop disease identification system.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.api.routes import dna_analysis, image_processing, disease_prediction
from src.api.database import init_db

# Initialize FastAPI app
app = FastAPI(
    title="DNA Crop Disease Identification API",
    description="API for DNA-based crop disease identification using computer vision and genetic analysis",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(dna_analysis.router, prefix="/api/dna", tags=["DNA Analysis"])
app.include_router(image_processing.router, prefix="/api/image", tags=["Image Processing"])
app.include_router(disease_prediction.router, prefix="/api/prediction", tags=["Disease Prediction"])

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    await init_db()

@app.get("/")
async def root():
    """Serve the web interface."""
    return FileResponse("static/index.html")

@app.get("/api")
async def api_info():
    """API information endpoint."""
    return {"message": "DNA Crop Disease Identification API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
