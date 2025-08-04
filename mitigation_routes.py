# mitigation_routes.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import io
import logging

from bias_mitigation import mitigate_bias

logger = logging.getLogger(__name__)

# Create router
mitigation_router = APIRouter()

# =============================
# Endpoint: POST /api/v1/mitigate-bias
# =============================
@mitigation_router.post("/mitigate-bias", summary="Mitigate Bias in AI Model")
async def mitigate_bias_endpoint(
    file: UploadFile = File(..., description="CSV file with model data"),
    sensitive_attribute: str = "gender",
    outcome: str = "loan_status",
    method: str = "threshold"  # or "exponentiated_gradient"
):
    """
    Upload a dataset and apply fairness mitigation.
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