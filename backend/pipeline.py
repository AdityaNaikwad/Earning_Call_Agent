# import os
# import re
# import json
# import asyncio
# import aiohttp
# import logging
# import tiktoken
# from dotenv import load_dotenv
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_groq import ChatGroq
# from langchain.schema import HumanMessage, SystemMessage

# load_dotenv()

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # ── Groq LLM ──
# llm = ChatGroq(
#     api_key=os.getenv("GROQ_API_KEY"),
#     model_name="meta-llama/llama-4-scout-17b-16e-instruct",
#     temperature=0.3,
#     request_timeout=60
# )


# def chunk_transcript(transcript: str) -> list[str]:
#     encoder = tiktoken.get_encoding("cl100k_base")
#     MAX_TOKENS = 450
#     OVERLAP_TOKENS = 50
#     sentences = re.split(r'(?<=[.!?])\s+', transcript)
#     chunks = []
#     current_chunk = []
#     current_token_count = 0

#     for sentence in sentences:
#         sentence = sentence.strip()
#         if not sentence:
#             continue

#         sentence_tokens = len(encoder.encode(sentence))

#         if sentence_tokens > MAX_TOKENS:
#             words = sentence.split()
#             word_chunk = []
#             word_token_count = 0
#             for word in words:
#                 word_tokens = len(encoder.encode(word))
#                 if word_token_count + word_tokens > MAX_TOKENS:
#                     if word_chunk:
#                         chunks.append(" ".join(word_chunk))
#                     word_chunk = [word]
#                     word_token_count = word_tokens
#                 else:
#                     word_chunk.append(word)
#                     word_token_count += word_tokens
#             if word_chunk:
#                 chunks.append(" ".join(word_chunk))
#             continue

#         if current_token_count + sentence_tokens > MAX_TOKENS:
#             if current_chunk:
#                 chunks.append(" ".join(current_chunk))
#             overlap_text = []
#             overlap_count = 0
#             for prev_sentence in reversed(current_chunk):
#                 prev_tokens = len(encoder.encode(prev_sentence))
#                 if overlap_count + prev_tokens > OVERLAP_TOKENS:
#                     break
#                 overlap_text.insert(0, prev_sentence)
#                 overlap_count += prev_tokens
#             current_chunk = overlap_text + [sentence]
#             current_token_count = overlap_count + sentence_tokens
#         else:
#             current_chunk.append(sentence)
#             current_token_count += sentence_tokens

#     if current_chunk:
#         chunks.append(" ".join(current_chunk))

#     logger.info(f"Token-based split: {len(chunks)} chunks")
#     return chunks


# # ── Async HuggingFace call for single chunk ──
# async def analyze_chunk(session: aiohttp.ClientSession, chunk: str):
#     try:
#         async with session.post(
#             "https://router.huggingface.co/hf-inference/models/ProsusAI/finbert",
#             headers={"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"},
#             json={"inputs": chunk[:512]},
#             timeout=aiohttp.ClientTimeout(total=15)
#         ) as response:
#             if response.status != 200:
#                 logger.warning(f"HF API error: {response.status}")
#                 return None
#             result = await response.json()
#             if not result or not isinstance(result, list):
#                 return None
#             chunk_scores = result[0] if isinstance(result[0], list) else result
#             top = max(chunk_scores, key=lambda x: x["score"])
#             return {"label": top["label"].lower(), "score": top["score"]}
#     except Exception as e:
#         logger.error(f"Chunk sentiment error: {str(e)}")
#         return None


# # ── Send chunks in batches of 5 ──
# async def analyze_sentiment_async(chunks: list[str]) -> dict:
#     scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
#     successful_chunks = 0
#     BATCH_SIZE = 5
#     BATCH_DELAY = 0.5
#     all_results = []

#     async with aiohttp.ClientSession() as session:
#         for i in range(0, len(chunks), BATCH_SIZE):
#             batch = chunks[i:i + BATCH_SIZE]
#             tasks = [analyze_chunk(session, chunk) for chunk in batch]
#             batch_results = await asyncio.gather(*tasks)
#             all_results.extend(batch_results)

#             if i + BATCH_SIZE < len(chunks):
#                 await asyncio.sleep(BATCH_DELAY)

#     for result in all_results:
#         if result and result["label"] in scores:
#             scores[result["label"]] += result["score"]
#             successful_chunks += 1

#     if successful_chunks == 0:
#         logger.warning("No chunks analyzed — defaulting to neutral")
#         return {
#             "scores": {"positive": 0.0, "negative": 0.0, "neutral": 100.0},
#             "overall": "neutral"
#         }

#     total = sum(scores.values())
#     for key in scores:
#         scores[key] = round(scores[key] / total * 100, 2)

#     overall = max(scores, key=scores.get)
#     logger.info(f"Sentiment scores: {scores}")
#     return {"scores": scores, "overall": overall}


# # ── Filter only financially relevant chunks ──
# def analyze_sentiment(chunks: list[str]) -> dict:
#     financial_keywords = [
#         "profit", "revenue", "NPA", "growth", "margin",
#         "crore", "percent", "%", "NIM", "CASA", "loan",
#         "deposit", "guidance", "outlook", "risk", "interest",
#         "income", "expense", "capital", "return", "asset"
#     ]

#     relevant_chunks = [
#         c for c in chunks
#         if any(kw.lower() in c.lower() for kw in financial_keywords)
#     ]

#     if len(relevant_chunks) < 5:
#         relevant_chunks = chunks

#     # Cap at 15 chunks max
#     relevant_chunks = relevant_chunks[:15]
#     logger.info(f"Analyzing {len(relevant_chunks)}/{len(chunks)} relevant chunks")

#     return asyncio.run(analyze_sentiment_async(relevant_chunks))


# def generate_report(
#     company: str,
#     transcript: str,
#     sentiment: dict
# ) -> dict:

#     company = company.strip().title()
#     logger.info(f"Sending full transcript: {len(transcript)} chars to LLaMA")

#     system_prompt = """You are a senior equity research analyst at a top Indian brokerage firm like Motilal Oswal or Kotak Securities. You have 15+ years of experience analyzing Indian PSU and private sector bank earnings calls.

# Analyze the COMPLETE earnings call transcript and generate a HIGHLY DETAILED structured report exactly like a professional research report.

# Respond ONLY in this exact JSON format:
# {
#   "summary": {
#     "macroeconomic_environment": "3-4 sentences on global and Indian macro environment discussed in the call",
#     "business_growth": "3-4 sentences with exact numbers on business growth, advances, deposits YoY",
#     "digital_initiatives": "2-3 sentences on digital products and tech initiatives mentioned",
#     "profitability": "3-4 sentences with exact figures on net profit, operating profit, NII, margins",
#     "deposits_advances_casa": "3-4 sentences with exact CASA ratio, deposit growth, advance growth numbers",
#     "guidance_outlook": "3-4 sentences on management guidance with specific growth targets for FY"
#   },
#   "key_highlights": [
#     "specific highlight with exact numbers e.g. Net profit surged 32% YoY to Rs 2,252 crore in Q1 FY26",
#     "specific highlight with exact numbers",
#     "specific highlight with exact numbers",
#     "specific highlight with exact numbers",
#     "specific highlight with exact numbers"
#   ],
#   "risks": {
#     "asset_quality_risk": "2-3 sentences on NPA, slippage risks with specific numbers",
#     "margin_nim_risk": "2-3 sentences on NIM compression, rate cut impact",
#     "operational_risk": "2-3 sentences on fraud, operational, cybersecurity risks",
#     "regulatory_risk": "2-3 sentences on ECL, RBI guidelines, compliance risks",
#     "liquidity_risk": "2-3 sentences on deposit concentration, funding risks",
#     "credit_cost_risk": "2-3 sentences on recovery uncertainty, provision coverage"
#   },
#   "signal": "Buy" or "Hold" or "Sell",
#   "signal_reason": "3-4 sentences with specific financial metrics justifying the signal"
# }

# Strict Rules:
# - Every section MUST contain actual numbers, percentages, crore figures from the transcript
# - Never write vague statements — always back with data from the call
# - Guidance section must include specific % targets management mentioned
# - Risks must be specific to THIS company not generic risks
# - Signal must reference actual financial performance metrics
# Return ONLY the JSON object. Absolutely no text before or after."""

#     user_prompt = f"""
# Company: {company}
# Overall Sentiment: {sentiment['overall']}
# Sentiment Breakdown: Positive {sentiment['scores']['positive']}% | Negative {sentiment['scores']['negative']}% | Neutral {sentiment['scores']['neutral']}%

# Full Earnings Call Transcript:
# {transcript}

# Generate the detailed analyst report now.
# """

#     messages = [
#         SystemMessage(content=system_prompt),
#         HumanMessage(content=user_prompt)
#     ]

#     try:
#         response = llm.invoke(messages)
#         raw = response.content.strip()
#         clean = raw.replace("```json", "").replace("```", "").strip()
#         report = json.loads(clean)
#         return report
#     except json.JSONDecodeError:
#         logger.error("LLM returned malformed JSON")
#         return {
#             "summary": {
#                 "macroeconomic_environment": "Could not generate report. Please try again.",
#                 "business_growth": "",
#                 "digital_initiatives": "",
#                 "profitability": "",
#                 "deposits_advances_casa": "",
#                 "guidance_outlook": ""
#             },
#             "key_highlights": [],
#             "risks": {
#                 "asset_quality_risk": "",
#                 "margin_nim_risk": "",
#                 "operational_risk": "",
#                 "regulatory_risk": "",
#                 "liquidity_risk": "",
#                 "credit_cost_risk": ""
#             },
#             "signal": "Hold",
#             "signal_reason": "Analysis unavailable"
#         }
#     except Exception as e:
#         logger.error(f"LLM call failed: {str(e)}")
#         raise ValueError(f"Report generation failed: {str(e)}")


# def run_pipeline(company: str, transcript: str) -> dict:
#     logger.info(f"Running pipeline for: {company}")
#     chunks = chunk_transcript(transcript)
#     sentiment = analyze_sentiment(chunks)
#     report = generate_report(company, transcript, sentiment)
#     return {
#         "company": company.strip().title(),
#         "sentiment": sentiment,
#         "report": report
#     }


# pipeline.py

import os
import re
import json
import asyncio
import aiohttp
import logging
import tiktoken
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Groq LLM ──
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.3,
    request_timeout=60
)


def chunk_transcript(transcript: str) -> list[str]:
    encoder = tiktoken.get_encoding("cl100k_base")
    MAX_TOKENS = 450
    OVERLAP_TOKENS = 50
    sentences = re.split(r'(?<=[.!?])\s+', transcript)
    chunks = []
    current_chunk = []
    current_token_count = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_tokens = len(encoder.encode(sentence))

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

        if current_token_count + sentence_tokens > MAX_TOKENS:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
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

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    logger.info(f"Token-based split: {len(chunks)} chunks")
    return chunks


# ── Async HuggingFace call for single chunk ──
async def analyze_chunk(session: aiohttp.ClientSession, chunk: str):
    try:
        async with session.post(
            "https://router.huggingface.co/hf-inference/models/ProsusAI/finbert",
            headers={"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"},
            json={"inputs": chunk[:512]},
            timeout=aiohttp.ClientTimeout(total=15)
        ) as response:
            if response.status != 200:
                logger.warning(f"HF API error: {response.status}")
                return None
            result = await response.json()
            if not result or not isinstance(result, list):
                return None
            chunk_scores = result[0] if isinstance(result[0], list) else result
            top = max(chunk_scores, key=lambda x: x["score"])
            return {"label": top["label"].lower(), "score": top["score"]}
    except Exception as e:
        logger.error(f"Chunk sentiment error: {str(e)}")
        return None


# ── Send chunks in batches of 5 ──
async def analyze_sentiment_async(chunks: list[str]) -> dict:
    scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    successful_chunks = 0
    BATCH_SIZE = 5
    BATCH_DELAY = 0.5
    all_results = []

    async with aiohttp.ClientSession() as session:
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            tasks = [analyze_chunk(session, chunk) for chunk in batch]
            batch_results = await asyncio.gather(*tasks)
            all_results.extend(batch_results)

            if i + BATCH_SIZE < len(chunks):
                await asyncio.sleep(BATCH_DELAY)

    for result in all_results:
        if result and result["label"] in scores:
            scores[result["label"]] += result["score"]
            successful_chunks += 1

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


# ── Filter only financially relevant chunks ──
def analyze_sentiment(chunks: list[str]) -> dict:
    financial_keywords = [
        "profit", "revenue", "NPA", "growth", "margin",
        "crore", "percent", "%", "NIM", "CASA", "loan",
        "deposit", "guidance", "outlook", "risk", "interest",
        "income", "expense", "capital", "return", "asset"
    ]

    relevant_chunks = [
        c for c in chunks
        if any(kw.lower() in c.lower() for kw in financial_keywords)
    ]

    if len(relevant_chunks) < 5:
        relevant_chunks = chunks

    # Cap at 15 chunks max
    relevant_chunks = relevant_chunks[:15]
    logger.info(f"Analyzing {len(relevant_chunks)}/{len(chunks)} relevant chunks")

    return asyncio.run(analyze_sentiment_async(relevant_chunks))


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