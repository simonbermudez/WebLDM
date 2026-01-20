"""
WebLDM API Routes

Provides REST API endpoints for:
- Data upload and preprocessing
- SVD analysis
- Global Analysis
- Lifetime Density Analysis (LDA)
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from ..analysis import Data, LDA, SVD_GA

router = APIRouter()

# Session storage for data objects (in production, use Redis or database)
sessions: Dict[str, Dict[str, Any]] = {}


# Pydantic models for request/response


class BoundsUpdate(BaseModel):
    session_id: str
    wl_lb: int
    wl_ub: int
    t0: int
    t: int


class IRFUpdate(BaseModel):
    session_id: str
    order: int = 1
    fwhm: float = 0.0
    munot: float = 0.0
    mus: List[float] = [0.0]
    lamnot: float = 500.0


class LDAParams(BaseModel):
    session_id: str
    tau_min: float = -1  # log10
    tau_max: float = 4  # log10
    n_taus: int = 100
    alpha_min: float = -4  # log10
    alpha_max: float = 2  # log10
    n_alphas: int = 20
    reg_method: str = "L2"  # L2, L1, elnet
    reg_matrix: str = "Id"  # Id, 1D, 2D, Fused
    simfit: bool = True
    ga_taus: Optional[List[float]] = None


class TSVDParams(BaseModel):
    session_id: str
    k: int = 5
    tau_min: float = -1
    tau_max: float = 4
    n_taus: int = 100
    ga_taus: Optional[List[float]] = None


class GAParams(BaseModel):
    session_id: str
    wLSVs: str = "3"  # Space-separated indices or single number
    initial_taus: List[float]
    bounds: List[List[float]]  # [[min, max], ...]
    alpha: float = 0.0
    fit_irf: bool = False


class ChirpFitRequest(BaseModel):
    session_id: str


# Helper functions


def get_session(session_id: str) -> Dict[str, Any]:
    """Get session data or raise error."""
    if session_id not in sessions:
        raise HTTPException(
            status_code=404, detail="Session not found. Please upload data first."
        )
    return sessions[session_id]


# Routes


@router.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    """
    Upload CSV data file.

    Expected format:
    - First row: wavelengths (first cell ignored)
    - First column: time points
    - Remaining cells: intensity data

    Returns session_id and data summary.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    content = await file.read()
    content_str = content.decode("utf-8")

    try:
        data = Data(content_str)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV: {str(e)}")

    # Generate session ID
    import uuid

    session_id = str(uuid.uuid4())

    # Store in session
    sessions[session_id] = {
        "data": data,
        "lda": None,
        "svd_ga": None,
        "filename": file.filename,
    }

    return {
        "session_id": session_id,
        "filename": file.filename,
        "info": data.get_info(),
        "raw_data": data.get_raw_data_plot(),
    }


@router.get("/data/{session_id}")
async def get_data_info(session_id: str):
    """Get data information and raw data plot."""
    session = get_session(session_id)
    data = session["data"]

    return {"info": data.get_info(), "raw_data": data.get_raw_data_plot()}


@router.post("/bounds")
async def update_bounds(params: BoundsUpdate):
    """Update data bounds (wavelength and time ranges)."""
    session = get_session(params.session_id)
    data = session["data"]

    try:
        data.update_bounds(params.wl_lb, params.wl_ub, params.t0, params.t)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error updating bounds: {str(e)}")

    return {"info": data.get_info(), "raw_data": data.get_raw_data_plot()}


@router.post("/irf")
async def update_irf(params: IRFUpdate):
    """Update IRF (Instrument Response Function) parameters."""
    session = get_session(params.session_id)
    data = session["data"]

    data.update_irf(params.order, params.fwhm, params.munot, params.mus, params.lamnot)

    return {
        "status": "ok",
        "irf": {
            "order": params.order,
            "fwhm": params.fwhm,
            "munot": params.munot,
            "mus": params.mus,
            "lamnot": params.lamnot,
        },
    }


@router.post("/chirp/fit")
async def fit_chirp(params: ChirpFitRequest):
    """Fit chirp to autocorrelation and apply correction."""
    session = get_session(params.session_id)
    data = session["data"]

    try:
        delay_shift, fitted_chirp = data.fit_chirp()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fitting chirp: {str(e)}")

    return {
        "wavelengths": data.wls_work.tolist(),
        "delay_shift": delay_shift.tolist(),
        "fitted_chirp": fitted_chirp.tolist(),
        "irf": {"munot": data.munot, "lamnot": data.lamnot, "mus": data.mu},
    }


@router.get("/svd/{session_id}")
async def get_svd(session_id: str):
    """Get SVD decomposition results."""
    session = get_session(session_id)
    data = session["data"]

    # Initialize SVD_GA if not already done
    if session["svd_ga"] is None:
        session["svd_ga"] = SVD_GA(data)

    svd_ga = session["svd_ga"]
    return svd_ga.get_svd_display()


@router.post("/ga")
async def run_global_analysis(params: GAParams):
    """Run Global Analysis to fit lifetimes."""
    session = get_session(params.session_id)
    data = session["data"]

    # Initialize SVD_GA if not already done
    if session["svd_ga"] is None:
        session["svd_ga"] = SVD_GA(data)
    else:
        session["svd_ga"].update_data(data)

    svd_ga = session["svd_ga"]

    # Convert bounds to list of tuples
    bounds = [tuple(b) for b in params.bounds]

    try:
        results = svd_ga.run_global_analysis(
            params.wLSVs, params.initial_taus, bounds, params.alpha, params.fit_irf
        )
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error in Global Analysis: {str(e)}"
        )

    return results


@router.post("/lda")
async def run_lda(params: LDAParams):
    """Run Lifetime Density Analysis."""
    session = get_session(params.session_id)
    data = session["data"]

    # Initialize LDA if not already done
    if session["lda"] is None:
        session["lda"] = LDA(data)
    else:
        session["lda"].update_data(data)

    lda = session["lda"]

    # Generate tau and alpha arrays
    taus = np.logspace(params.tau_min, params.tau_max, params.n_taus)
    alphas = np.logspace(params.alpha_min, params.alpha_max, params.n_alphas)

    # Create regularization matrix
    L = LDA.create_regularization_matrix(len(taus), params.reg_matrix)

    # Update parameters
    lda.update_params(taus, alphas, params.reg_method, L, params.simfit)

    try:
        results = lda.run_lda(params.ga_taus)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in LDA: {str(e)}")

    return results


@router.post("/tsvd")
async def run_tsvd(params: TSVDParams):
    """Run Truncated SVD regularization."""
    session = get_session(params.session_id)
    data = session["data"]

    # Initialize LDA if not already done
    if session["lda"] is None:
        session["lda"] = LDA(data)
    else:
        session["lda"].update_data(data)

    lda = session["lda"]

    try:
        results = lda.run_tsvd(
            params.k, params.tau_min, params.tau_max, params.n_taus, params.ga_taus
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in TSVD: {str(e)}")

    return results


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and free memory."""
    if session_id in sessions:
        del sessions[session_id]
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


@router.get("/sessions")
async def list_sessions():
    """List all active sessions (for debugging)."""
    return {
        "sessions": [
            {"id": sid, "filename": s.get("filename", "unknown")}
            for sid, s in sessions.items()
        ]
    }
