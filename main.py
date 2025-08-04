# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from audit_routes import audit_router
from mitigation_routes import mitigation_router

# =============================
# Initialize App
# =============================
app = FastAPI(
    title="FairLens API",
    description="An AI Bias Auditor & Mitigator for ethical AI in hiring and lending",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# =============================
# Add CORS Middleware
# =============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# Health Check
# =============================
@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "message": "FairLens API is running!"}

# =============================
# Include Routers
# =============================
app.include_router(audit_router, prefix="/api/v1", tags=["Bias Audit"])
app.include_router(mitigation_router, prefix="/api/v1", tags=["Bias Mitigation"])

# =============================
# Root Info
# =============================
@app.get("/", tags=["Root"])
def home():
    return {
        "app": "FairLens API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "bias_audit": "/api/v1/audit",
            "bias_mitigation": "/api/v1/mitigate-bias",
            "health": "/health"
        }
    }