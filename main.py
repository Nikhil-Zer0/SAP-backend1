# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import io
import logging

# Import your core audit logic
from fairlens_core import run_bias_audit
# Import your bias mitigation logic
from bias_mitigation import mitigate_bias

# Initialize app
app = FastAPI(
    title="FairLens API",
    description="An AI Bias Auditor for hiring and loan models",
    version="1.0.0"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable CORS (allow frontend to call API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
def home():
    return {
        "message": "Welcome to FairLens API",
        "docs": "/docs",
        "endpoints": {
            "POST /audit": "Upload CSV to audit AI bias"
        }
    }

# Audit endpoint
@app.post("/audit")
async def audit_bias(
    file: UploadFile = File(..., description="CSV file with model data"),
    sensitive_attribute: str = "gender",
    outcome: str = "loan_status"
):
    """
    Upload a CSV file to audit for bias in AI/ML decisions.
    Returns a structured bias report.
    """
    # Validate file type
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        # Read file
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))

        logger.info(f"Received file: {file.filename}, shape: {df.shape}")

        # Run bias audit
        report = run_bias_audit(
            df=df,
            sensitive_attribute=sensitive_attribute,
            outcome=outcome
        )

        # Handle errors from core
        if report.get("error"):
            logger.error(f"Audit failed: {report['error']}")
            raise HTTPException(status_code=500, detail=report["error"])

        return JSONResponse(content=report)

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Invalid CSV format.")
    except Exception as e:
        logger.exception("Unexpected error during audit")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
# =========================
# New Endpoint: /mitigate-bias
# =========================
@app.post("/mitigate-bias")
async def mitigate_bias_endpoint(
    file: UploadFile = File(...),
    sensitive_attribute: str = "gender",
    outcome: str = "loan_status",
    method: str = "threshold"  # or "exponentiated_gradient"
):
    """
    Upload a dataset and apply bias mitigation.
    Returns fairness metrics before and after.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
        logger.info(f"Mitigation: Received {file.filename}, shape: {df.shape}")

        result = mitigate_bias(
            df=df,
            sensitive_attribute=sensitive_attribute,
            outcome=outcome,
            method=method
        )

        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result["message"])

        return JSONResponse(content=result)

    except Exception as e:
        logger.exception("Mitigation failed")
        raise HTTPException(status_code=500, detail=f"Mitigation failed: {str(e)}")
    
@app.get("/health")
def health():
    return {"status": "ok", "modules": ["fairlens_core", "bias_mitigator"]}