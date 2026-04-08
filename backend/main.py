# main.py

import os
import logging
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv
from scraper import extract_transcript_from_pdf
from pipeline import run_pipeline

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

# ── Rate limiter ──
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Earnings Call Analyst Agent")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS ──
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health check ──
@app.get("/")
def root():
    return {"status": "Earnings Call Analyst Agent is running"}


# ── Main endpoint ──
@app.post("/analyze")
@limiter.limit("5/minute")
async def analyze(
    request: Request,
    company: str = Form(...),
    file: UploadFile = File(...)
):
    logger.info(f"Received request for company: {company}")

    try:
        # Validate file type
        if not file.filename.endswith(".pdf"):
            return JSONResponse(
                status_code=400,
                content={"error": "Only PDF files are accepted."}
            )

        # Validate company name
        if not company.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "Company name cannot be empty."}
            )

        # Read PDF
        pdf_bytes = await file.read()
        logger.info(f"PDF received: {file.filename} ({len(pdf_bytes)} bytes)")

        # Extract transcript
        transcript = extract_transcript_from_pdf(pdf_bytes)

        # Run pipeline
        result = run_pipeline(company, transcript)

        logger.info(f"Pipeline complete for: {company}")
        return JSONResponse(status_code=200, content=result)

    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        logger.error(f"Internal error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error. Please try again."}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)