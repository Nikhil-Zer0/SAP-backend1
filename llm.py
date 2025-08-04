from fastapi import APIRouter, HTTPException, Body, File, UploadFile
from pydantic import BaseModel
from typing import Dict, Any, List
import google.generativeai as genai
import os
import json
import io
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create router
router = APIRouter()

def get_api_key():
    """Get API key from environment variable or fallback to hardcoded value"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        # Fallback to your hardcoded key (not recommended for production)
        api_key = "AIzaSyAm6y0UeH0gI_E3fDJ0lcVUZARW8_rxq-c"
    return api_key

def configure_gemini():
    """Configure Gemini with API key"""
    try:
        api_key = get_api_key()
        if not api_key:
            raise RuntimeError("No Google API key found")
        
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.5-pro")  # Updated to stable model name
    except Exception as e:
        raise RuntimeError(f"Failed to configure Gemini: {str(e)}")

class FairnessAnalysisInput(BaseModel):
    summary: Dict[str, Any]
    fairness_metrics: Dict[str, float]
    refutations: Dict[str, Any]
    explainability: Dict[str, float]
    recommendation: str
    status: str

@router.post("/analyze_fairness_results")
async def analyze_fairness_results(
    analysis_data: FairnessAnalysisInput = Body(...)
):
    """
    Analyze fairness metrics results with Gemini AI for deeper interpretation
    """
    try:
        # Configure Gemini model
        model = configure_gemini()
        
        # Convert input data to dict for easier formatting
        data_dict = analysis_data.dict()
        
        # Create comprehensive prompt for analysis
        prompt = f"""
        You are a fairness and explainability expert analyzing machine learning model fairness results.
        
        Below are the fairness analysis results from a previous evaluation:
        
        {json.dumps(data_dict, indent=2, default=str)}
        
        Please provide a comprehensive interpretation covering:
        
        1. **Results Interpretation**:
           - Explain what each metric means in practical terms
           - Highlight the most concerning findings
           - Put the effect sizes in context
        
        2. **Significance Assessment**:
           - Analyze the confidence intervals and significance
           - Explain what the refutation tests tell us
           - Assess the robustness of the findings
        
        3. **Root Cause Analysis**:
           - Potential reasons for the observed biases
           - Data or modeling issues that might contribute
        
        4. **Actionable Recommendations**:
           - Specific mitigation strategies
           - Model or data improvements
           - Any necessary policy changes
        
        5. **Risk Assessment**:
           - Potential real-world impacts
           - Legal or ethical considerations
           - Urgency of addressing the issues
        
        Provide clear, professional analysis with concrete examples and explanations suitable for both technical and non-technical stakeholders.
        """

        # Generate analysis with Gemini
        try:
            response = model.generate_content(prompt)
            analysis_text = response.text
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Error generating analysis with Gemini: {str(e)}"
            )
        
        return {
            "status": "success",
            "original_results": data_dict,
            "gemini_analysis": analysis_text
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Unexpected error during analysis: {str(e)}"
        )

class LoanQueryInput(BaseModel):
    search_criteria: Dict[str, Any]  # Search criteria to find the specific loan application
    question: str  # User's question about the loan decision
    analysis_results: FairnessAnalysisInput  # Fairness analysis results

class GeminiRequest(BaseModel):
    model_output: Dict[str, Any]
    user_prompt: str

@router.post("/analyze_loan_decision")
async def analyze_loan_decision(payload: GeminiRequest = Body(...)):
    try:
        # Load dataset
        try:
            df = pd.read_csv('dataset/loan.csv')
            dataset = df.to_dict(orient='records')
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading dataset: {str(e)}")

        # Configure Gemini model
        model = configure_gemini()

        # Create prompt to send to Gemini
        prompt = f"""
You are an AI assistant analyzing fairness and decision patterns in loan approvals.

Here is the full dataset of loan applicants:
{json.dumps(dataset, indent=2)}

Here are the model outputs from fairness and causal analysis:
{json.dumps(payload.model_output, indent=2)}

Here is the user's question:
"{payload.user_prompt}"

Instructions:
- Answer clearly and professionally.
- Analyze any bias or unfairness if present.
- Use data and metrics provided to justify your conclusions.
- Provide recommendations or next steps if applicable.
"""

        # Generate response
        try:
            gemini_response = model.generate_content(prompt)
            return {
                "status": "success",
                "response": gemini_response.text
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# Health check endpoint (unchanged)
@router.get("/health")
async def health_check():
    """Check if the service is running and Gemini is configured"""
    try:
        model = configure_gemini()
        return {
            "status": "healthy",
            "gemini_configured": True,
            "message": "Service is running and Gemini is configured"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "gemini_configured": False,
            "error": str(e)
        }