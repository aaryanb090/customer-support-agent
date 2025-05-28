import os
import glob
import csv
import faiss
import warnings
import requests
import logging
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline, logging as hf_logging
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from datetime import datetime

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=".*xla_device.*")

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_API_TOKEN:
    logger.error("HF_API_TOKEN not set")
    raise RuntimeError("Please set the HF_API_TOKEN environment variable")

GREETINGS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}
def is_greeting(text: str) -> bool:
    s = text.lower().strip()
    return any(s.startswith(g) for g in GREETINGS)

ESCALATE_KEYWORDS = {
    "refund", "money back", "billing error", "overcharged", "cancel subscription",
    "crash", "bug", "error", "outage", "delay", "stuck",
    "data breach", "security", "hack", "delete account", "lost data"
}

SUPPORT_RE = re.compile(
    r"\b(reset|forgot|change|update)\b.*\b(password|pwd|profile|avatar|pic(?:ture)?)\b",
    re.I
)
SALES_RE = re.compile(r"\b(price|pricing|cost|quote|plan|subscription|tier)\b", re.I)
FEATURE_RE = re.compile(r"\b(dark[\s\-]?mode|feature|theme)\b", re.I)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ZS_LABELS = [
    "technical support request",
    "product feature suggestion",
    "sales inquiry",
    "small talk / general inquiry"
]
LABEL_MAP = {
    "technical support request":   "Technical Support",
    "product feature suggestion":   "Product Feature Request",
    "sales inquiry":                "Sales Lead",
    "small talk / general inquiry": "Other"
}

sentiment = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

KB_DIR = os.path.join(os.path.dirname(__file__), "kb")
paths = glob.glob(os.path.join(KB_DIR, "*.*"))
chunks: List[str] = []
meta: List[Dict] = []
for p in paths:
    fn = os.path.basename(p)
    txt = open(p, "r", encoding="utf-8").read().strip()
    paras = [s.strip() for s in txt.split("\n\n") if s.strip()] or [txt]
    for seg in paras:
        chunks.append(seg)
        meta.append({"doc": fn, "text": seg})

if chunks:
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    emb = emb.astype("float32"); faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb)
    logger.info(f"Built KB index with {index.ntotal} chunks")
else:
    index = None
    embed_model = None

FEATURE_LOG = os.path.join(os.path.dirname(__file__), "feature_requests.csv")
if not os.path.exists(FEATURE_LOG):
    with open(FEATURE_LOG, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["timestamp", "user_id", "message"])

MISSING_LOG = os.path.join(os.path.dirname(__file__), "missing_docs.csv")
if not os.path.exists(MISSING_LOG):
    with open(MISSING_LOG, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["timestamp", "user_id", "message"])

class ChatRequest(BaseModel):
    user_id: str
    message: str
    top_k: int = 3

class KBResult(BaseModel):
    doc: str
    text: str
    score: float

class AgentResponse(BaseModel):
    intent: str
    confidence: float
    answer: str
    kb_results: Optional[List[KBResult]] = None
    escalate: bool

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/agent", response_model=AgentResponse)
async def agent_endpoint(req: ChatRequest):
    text  = req.message.strip()
    lower = text.lower()
    logger.debug(f"Received {text!r} from {req.user_id!r}")

    if is_greeting(text):
        return AgentResponse(
            intent="Greeting",
            confidence=1.0,
            answer="Hello there! 👋 How can I assist you today?",
            kb_results=None,
            escalate=False
        )

    if len(text) < 3:
        return AgentResponse(
            intent="Other",
            confidence=1.0,
            answer="Could you please elaborate a bit more?",
            kb_results=None,
            escalate=False
        )

    if any(k in lower for k in ESCALATE_KEYWORDS):
        return AgentResponse(
            intent="Technical Support",
            confidence=1.0,
            answer="I’m sorry you’re having trouble. I’m connecting you to a live agent now.",
            kb_results=None,
            escalate=True
        )

    if SUPPORT_RE.search(text):
        intent, confidence = "Technical Support", 1.0
    elif SALES_RE.search(text):
        return AgentResponse(
            intent="Sales Lead",
            confidence=1.0,
            answer="Thank you for your interest! Our sales team will be in touch shortly to learn more about your needs.",
            kb_results=None,
            escalate=False
        )
    elif FEATURE_RE.search(text):
        with open(FEATURE_LOG, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([datetime.utcnow().isoformat(), req.user_id, text])
        return AgentResponse(
            intent="Product Feature Request",
            confidence=1.0,
            answer="Thank you for your suggestion! We’ve logged your feature request for review.",
            kb_results=None,
            escalate=False
        )
    else:
        hf_url  = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        payload = {"inputs": text, "parameters": {"candidate_labels": ZS_LABELS}}
        r = requests.post(hf_url, headers=headers, json=payload)
        r.raise_for_status()
        resp = r.json()

        if "labels" in resp:
            lst = [{"label": lab, "score": sc}
                   for lab, sc in zip(resp["labels"], resp["scores"])]
        else:
            lst = resp

        top       = max(lst, key=lambda x: x["score"])
        raw_label = top["label"]
        intent    = LABEL_MAP.get(raw_label, "Other")
        confidence= float(top["score"])

        if raw_label == "small talk / general inquiry":
            return AgentResponse(
                intent="Other",
                confidence=confidence,
                answer="I’m sorry, I didn’t quite catch that. Could you please rephrase?",
                kb_results=None,
                escalate=False
            )

        if intent == "Other":
            return AgentResponse(
                intent="Other",
                confidence=confidence,
                answer=(
                    "I’m sorry, I didn’t quite catch that. "
                    "I can help with technical support, feature requests, or sales inquiries—"
                    "would you like to speak to a live agent?"
                ),
                kb_results=None,
                escalate=False
            )

    kb_recs: List[KBResult] = []
    if index:
        qv = embed_model.encode([text], convert_to_numpy=True, show_progress_bar=False)
        if qv.ndim == 1:
            qv = qv.reshape(1, -1)
        qv = qv.astype("float32"); faiss.normalize_L2(qv)
        scores, ids = index.search(qv, req.top_k)

        for sc, i in zip(scores[0], ids[0]):
            kb_recs.append(
                KBResult(doc=meta[i]["doc"], text=meta[i]["text"], score=float(sc))
            )

    valid_recs = [r for r in kb_recs if r.score >= 0.2]

    if valid_recs:
        domain_keywords = re.findall(
            r'password|pwd|profile|avatar|pic(?:ture)?',
            text.lower()
        )
        if domain_keywords:
            filtered = [
                r for r in valid_recs
                if any(kw in r.text.lower() for kw in domain_keywords)
            ]
        else:
            filtered = valid_recs

        to_use = filtered if filtered else valid_recs[: req.top_k]

        bullets = "\n".join(f"{i+1}. {r.text}" for i, r in enumerate(to_use))

        answer = (
            "Thank you for reaching out. Based on our documentation, here are the steps:\n\n"
            f"{bullets}\n\n"
            "Did this solve your issue?"
        )
        return AgentResponse(
            intent="Technical Support",
            confidence=confidence,
            answer=answer,
            kb_results=to_use,
            escalate=False
        )

    sent = sentiment(text)[0]
    if sent["label"] == "NEGATIVE" and sent["score"] > 0.7:
        return AgentResponse(
            intent="Technical Support",
            confidence=confidence,
            answer="I’m sorry you’re having trouble. I’m connecting you to a live agent now.",
            kb_results=None,
            escalate=True
        )

    with open(MISSING_LOG, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([datetime.utcnow().isoformat(), req.user_id, text])

    return AgentResponse(
        intent="Technical Support",
        confidence=confidence,
        answer=(
            "Thanks for your query. I couldn't find an immediate answer, "
            "but I've routed your request to our relevant department team. "
            "They will get back to you shortly."
        ),
        kb_results=None,
        escalate=True
    )
