import os
import re
import json
import logging
import requests
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from huggingface_hub import InferenceClient

# Replace old HF_API_URL and HF_HEADERS with this
hf_client = InferenceClient(
    model="ProsusAI/finbert",
    token=os.getenv("HF_API_KEY")
)
# ── Groq LLM ──
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.3,
    request_timeout=30
)


import tiktoken

def chunk_transcript(transcript: str) -> list[str]:
    """
    Token-based chunking using tiktoken.
    Respects sentence boundaries for better FinBERT accuracy.
    """
    # Use cl100k_base encoder (same as GPT-4, works well for finance)
    encoder = tiktoken.get_encoding("cl100k_base")

    # FinBERT max = 512 tokens
    MAX_TOKENS = 450      # leave buffer for special tokens
    OVERLAP_TOKENS = 50   # overlap so context isn't lost between chunks

    # First split into sentences to respect boundaries
    sentences = re.split(r'(?<=[.!?])\s+', transcript)

    chunks = []
    current_chunk = []
    current_token_count = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Count tokens in this sentence
        sentence_tokens = len(encoder.encode(sentence))

        # If single sentence exceeds limit, split it further by words
        if sentence_tokens > MAX_TOKENS:
            words = sentence.split()
            word_chunk = []
            word_token_count = 0
            for word in words:
                word_tokens = len(encoder.encode(word))
                if word_token_count + word_tokens > MAX_TOKENS:
                    if word_chunk:
                        chunks.append(" ".join(word_chunk))
                    word_chunk = [word]
                    word_token_count = word_tokens
                else:
                    word_chunk.append(word)
                    word_token_count += word_tokens
            if word_chunk:
                chunks.append(" ".join(word_chunk))
            continue

        # If adding this sentence exceeds limit — save chunk, start new
        if current_token_count + sentence_tokens > MAX_TOKENS:
            if current_chunk:
                chunks.append(" ".join(current_chunk))

            # Overlap — keep last few sentences for context continuity
            overlap_text = []
            overlap_count = 0
            for prev_sentence in reversed(current_chunk):
                prev_tokens = len(encoder.encode(prev_sentence))
                if overlap_count + prev_tokens > OVERLAP_TOKENS:
                    break
                overlap_text.insert(0, prev_sentence)
                overlap_count += prev_tokens

            current_chunk = overlap_text + [sentence]
            current_token_count = overlap_count + sentence_tokens
        else:
            current_chunk.append(sentence)
            current_token_count += sentence_tokens

    # Don't forget last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    logger.info(f"Token-based split: {len(chunks)} chunks")
    return chunks


def analyze_sentiment(chunks: list[str]) -> dict:
    scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    successful_chunks = 0

    for chunk in chunks:
        try:
            result = hf_client.text_classification(chunk[:512])
            # result is a list of ClassificationOutput objects
            top = max(result, key=lambda x: x.score)
            label = top.label.lower()
            if label in scores:
                scores[label] += top.score
                successful_chunks += 1

        except Exception as e:
            logger.error(f"Sentiment error on chunk: {str(e)}")
            continue

    if successful_chunks == 0:
        logger.warning("No chunks analyzed — defaulting to neutral")
        return {
            "scores": {"positive": 0.0, "negative": 0.0, "neutral": 100.0},
            "overall": "neutral"
        }

    total = sum(scores.values())
    for key in scores:
        scores[key] = round(scores[key] / total * 100, 2)

    overall = max(scores, key=scores.get)
    logger.info(f"Sentiment scores: {scores}")
    return {"scores": scores, "overall": overall}


def smart_truncate(transcript: str, max_chars: int = 4000) -> str:
    if len(transcript) <= max_chars:
        return transcript
    # Cut at sentence boundary
    sentences = re.split(r'(?<=[.!?])\s+', transcript)
    snippet = ""
    for s in sentences:
        if len(snippet) + len(s) > max_chars:
            break
        snippet += s + " "
    return snippet.strip()


def generate_report(
    company: str,
    transcript: str,
    sentiment: dict
) -> dict:

    company = company.strip().title()
    logger.info(f"Sending full transcript: {len(transcript)} chars to LLaMA")

    system_prompt = """You are a senior equity research analyst at a top Indian brokerage firm like Motilal Oswal or Kotak Securities. You have 15+ years of experience analyzing Indian PSU and private sector bank earnings calls.

Analyze the COMPLETE earnings call transcript and generate a HIGHLY DETAILED structured report exactly like a professional research report.

Respond ONLY in this exact JSON format:
{
  "summary": {
    "macroeconomic_environment": "3-4 sentences on global and Indian macro environment discussed in the call",
    "business_growth": "3-4 sentences with exact numbers on business growth, advances, deposits YoY",
    "digital_initiatives": "2-3 sentences on digital products and tech initiatives mentioned",
    "profitability": "3-4 sentences with exact figures on net profit, operating profit, NII, margins",
    "deposits_advances_casa": "3-4 sentences with exact CASA ratio, deposit growth, advance growth numbers",
    "guidance_outlook": "3-4 sentences on management guidance with specific growth targets for FY"
  },
  "key_highlights": [
    "specific highlight with exact numbers e.g. Net profit surged 32% YoY to Rs 2,252 crore in Q1 FY26",
    "specific highlight with exact numbers",
    "specific highlight with exact numbers",
    "specific highlight with exact numbers",
    "specific highlight with exact numbers"
  ],
  "risks": {
    "asset_quality_risk": "2-3 sentences on NPA, slippage risks with specific numbers",
    "margin_nim_risk": "2-3 sentences on NIM compression, rate cut impact",
    "operational_risk": "2-3 sentences on fraud, operational, cybersecurity risks",
    "regulatory_risk": "2-3 sentences on ECL, RBI guidelines, compliance risks",
    "liquidity_risk": "2-3 sentences on deposit concentration, funding risks",
    "credit_cost_risk": "2-3 sentences on recovery uncertainty, provision coverage"
  },
  "signal": "Buy" or "Hold" or "Sell",
  "signal_reason": "3-4 sentences with specific financial metrics justifying the signal"
}

Strict Rules:
- Every section MUST contain actual numbers, percentages, crore figures from the transcript
- Never write vague statements — always back with data from the call
- If a section topic was not discussed, write what WAS discussed instead
- Guidance section must include specific % targets management mentioned
- Risks must be specific to THIS company not generic risks
- Signal must reference actual financial performance metrics
Return ONLY the JSON object. Absolutely no text before or after."""

    user_prompt = f"""
Company: {company}
Overall Sentiment: {sentiment['overall']}
Sentiment Breakdown: Positive {sentiment['scores']['positive']}% | Negative {sentiment['scores']['negative']}% | Neutral {sentiment['scores']['neutral']}%

Full Earnings Call Transcript:
{transcript}

Generate the detailed analyst report now.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()
        clean = raw.replace("```json", "").replace("```", "").strip()
        report = json.loads(clean)
        return report
    except json.JSONDecodeError:
        logger.error("LLM returned malformed JSON")
        return {
            "summary": {
                "macroeconomic_environment": "Could not generate report. Please try again.",
                "business_growth": "",
                "digital_initiatives": "",
                "profitability": "",
                "deposits_advances_casa": "",
                "guidance_outlook": ""
            },
            "key_highlights": [],
            "risks": {
                "asset_quality_risk": "",
                "margin_nim_risk": "",
                "operational_risk": "",
                "regulatory_risk": "",
                "liquidity_risk": "",
                "credit_cost_risk": ""
            },
            "signal": "Hold",
            "signal_reason": "Analysis unavailable"
        }
    except Exception as e:
        logger.error(f"LLM call failed: {str(e)}")
        raise ValueError(f"Report generation failed: {str(e)}")


def run_pipeline(company: str, transcript: str) -> dict:
    logger.info(f"Running pipeline for: {company}")

    chunks = chunk_transcript(transcript)
    sentiment = analyze_sentiment(chunks)
    report = generate_report(company, transcript, sentiment)

    return {
        "company": company.strip().title(),
        "sentiment": sentiment,
        "report": report
    }