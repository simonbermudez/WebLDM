"""
WebLDM - Web-based Lifetime Density Analysis

FastAPI application for performing lifetime density analysis
on time-resolved spectroscopic data.

Based on PyLDM by Gabriel Dorlhiac and Clyde Fare.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router

app = FastAPI(
    title="WebLDM",
    description="Web-based Lifetime Density Analysis for time-resolved spectroscopy",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "WebLDM API",
        "version": "1.0.0",
        "description": "Web-based Lifetime Density Analysis",
        "docs": "/api/docs",
        "endpoints": {
            "upload": "POST /api/upload - Upload CSV data",
            "data": "GET /api/data/{session_id} - Get data info",
            "bounds": "POST /api/bounds - Update data bounds",
            "irf": "POST /api/irf - Update IRF parameters",
            "chirp": "POST /api/chirp/fit - Fit chirp correction",
            "svd": "GET /api/svd/{session_id} - Get SVD results",
            "ga": "POST /api/ga - Run Global Analysis",
            "lda": "POST /api/lda - Run Lifetime Density Analysis",
            "tsvd": "POST /api/tsvd - Run Truncated SVD",
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}
