# audit_routes.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import io
import logging

from fairlens_core import run_bias_audit

logger = logging.getLogger(__name__)

# Create router
audit_router = APIRouter()

# =============================
# Endpoint: POST /api/v1/audit
# =============================
@audit_router.post("/audit", summary="Audit AI Model for Bias")
async def audit_bias(
    file: UploadFile = File(..., description="CSV file with model data"),
    sensitive_attribute: str = "gender",
    outcome: str = "loan_status"
):
    """
    Upload a dataset to detect bias in AI decisions.
    Returns a structured bias audit report.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
        logger.info(f"Audit: Received {file.filename}, shape: {df.shape}")

        report = run_bias_audit(
            df=df,
            sensitive_attribute=sensitive_attribute,
            outcome=outcome
        )

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