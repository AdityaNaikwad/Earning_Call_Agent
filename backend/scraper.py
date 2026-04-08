import io
import logging
import pdfplumber

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MIN_TEXT_LENGTH = 100


def extract_transcript_from_pdf(pdf_bytes: bytes) -> str:
    # Validate file size
    if len(pdf_bytes) > MAX_FILE_SIZE:
        raise ValueError("File too large. Maximum size is 10MB.")

    text = ""
    pdf_file = io.BytesIO(pdf_bytes)

    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        logger.error(f"PDF parsing failed: {str(e)}")
        raise ValueError("Could not parse PDF. Make sure it is a valid PDF file.")

    if len(text.strip()) < MIN_TEXT_LENGTH:
        raise ValueError(
            "Could not extract text. PDF may be scanned or image-based."
        )

    logger.info(f"Extracted {len(text)} characters from PDF")
    return text.strip()