import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv

from config.config import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

try:
    import ollama
    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False

try:
    from groq import Groq
    _GROQ_SDK_AVAILABLE = True
except ImportError:
    _GROQ_SDK_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# LLM backend selection. Groq (hosted) is what runs in deployment — no local
# GPU/server to manage. Ollama stays as the local-dev fallback: if
# GROQ_API_KEY isn't set (e.g. a laptop without a Groq account), the pipeline
# keeps working exactly as before, unchanged. LLM_BACKEND lets either be
# forced explicitly (e.g. LLM_BACKEND=ollama to test the local path even
# with a key configured).
# ─────────────────────────────────────────────────────────────────────────────
_GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
_LLM_BACKEND_OVERRIDE = os.environ.get("LLM_BACKEND", "").strip().lower()

if _LLM_BACKEND_OVERRIDE == "groq":
    _USE_GROQ = True
elif _LLM_BACKEND_OVERRIDE == "ollama":
    _USE_GROQ = False
else:
    _USE_GROQ = bool(_GROQ_API_KEY) and _GROQ_SDK_AVAILABLE

_groq_client = Groq(api_key=_GROQ_API_KEY) if (_USE_GROQ and _GROQ_API_KEY) else None

_GROQ_MODEL = "openai/gpt-oss-20b"
# NOTE: as of 2026-09, Groq's catalog no longer serves any Llama model —
# llama-3.1-8b-instant returns 404 model_not_found (confirmed live via
# client.models.list()). gpt-oss-20b is the closest fast/small equivalent
# on their current lineup. Re-check client.models.list() if this 404s again.
_OLLAMA_MODEL_PRIMARY = "llama3.1:8b"
_OLLAMA_MODEL_FALLBACK = "llama3:latest"    # used when primary is OOM-killed

SYSTEM_PROMPT = """You are a bioinformatics assistant helping scientists at \
AstraZeneca select cell lines for experiments. You are given structured \
evidence about a cell line's suitability for studying a specific gene. \
Your job is to:
1. Explain WHY this cell line is or isn't suitable
2. Highlight the strongest evidence
3. Flag any trade-offs or concerns
4. Suggest what type of experiment it suits best
Be concise, precise, and scientifically accurate. Use plain English that \
a bench scientist can act on. Never make up data — only use what is provided. \
When the evidence states which data sources are missing, report exactly those \
— do not soften, generalise, or invent data coverage."""


def _chat_groq(prompt: str) -> str:
    """Send a prompt to Groq's hosted Llama 3.1 8B and return the response text."""
    if _groq_client is None:
        raise RuntimeError(
            "Groq backend selected but not usable — GROQ_API_KEY missing or "
            "the groq package isn't installed (pip install groq)."
        )
    try:
        response = _groq_client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            # 2048, not 1024: gpt-oss-20b is a reasoning model that spends
            # part of the token budget on internal reasoning before the
            # visible 5-section answer — 1024 was observed to truncate
            # mid-way through section 4 in testing.
            max_tokens=2048,
        )
    except Exception as exc:
        raise RuntimeError(f"Groq API call failed: {exc}") from exc
    return response.choices[0].message.content


def _chat_ollama(prompt: str) -> str:
    """Send a prompt to a local ollama server and return the response text.

    Tries _OLLAMA_MODEL_PRIMARY first; if the model is OOM-killed falls back
    to _OLLAMA_MODEL_FALLBACK automatically (llama4:scout needs ~67 GB RAM).
    """
    if not _OLLAMA_AVAILABLE:
        raise RuntimeError(
            "ollama Python package not installed. Run: pip install ollama"
        )

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]

    for model in (_OLLAMA_MODEL_PRIMARY, _OLLAMA_MODEL_FALLBACK):
        try:
            response = ollama.chat(model=model, messages=msgs)
            if model != _OLLAMA_MODEL_PRIMARY:
                print(f"  [generator] Using fallback model: {model}")
            return response["message"]["content"]
        except Exception as exc:
            err = str(exc)
            if "connection" in err.lower() or "refused" in err.lower():
                raise ConnectionError(
                    "Cannot reach ollama. Start it with: ollama serve"
                ) from exc
            if "killed" in err.lower() or "500" in err or "terminated" in err.lower():
                # OOM or crash — try next model
                print(f"  [generator] {model} failed (likely OOM), trying fallback...")
                continue
            raise

    raise RuntimeError(
        f"All models failed. Primary: {_OLLAMA_MODEL_PRIMARY}, "
        f"Fallback: {_OLLAMA_MODEL_FALLBACK}"
    )


def _chat(prompt: str) -> str:
    """
    Route to Groq (hosted, used in deployment) or local Ollama (dev
    fallback when GROQ_API_KEY isn't configured) — decided once at import
    time by _USE_GROQ. Prompt construction (SYSTEM_PROMPT, the caller's
    prompt text) is identical either way; only the API call differs.
    """
    if _USE_GROQ:
        return _chat_groq(prompt)
    return _chat_ollama(prompt)


def generate_justification(
    gene: str,
    evidence_context: str,
    cell_line_name: str,
) -> str:
    """
    Generate a structured justification for whether a cell line is suitable
    for studying the given gene.
    """
    prompt = f"""Based on this evidence, justify whether {cell_line_name} is a \
good choice for studying {gene}:

{evidence_context}

Provide your answer in exactly this format:
1. RECOMMENDATION: (Strongly Recommended / Recommended / Use with Caution / Not Recommended)
2. KEY REASON: (one sentence — the single most important factor)
3. EVIDENCE SUMMARY: (2-3 sentences on the expression data)
4. TRADE-OFFS: (concerns and limitations, in prose. Your first sentence must \
name the data sources from the "MISSING SOURCES" line of the DATA COVERAGE \
section — those exact source names and only those, written into a normal \
sentence (do NOT copy the bracketed [MISSING]/[PRESENT] list). If that line \
says "none", write "All five evidence sources have data for this pair." \
In the TRADE-OFFS section, also state the mutation status exactly as given \
in the MUTATION STATUS block above — whether a damaging variant was found \
and cited, whether the gene is confirmed wild-type (variant calls exist, \
none damaging), or whether mutation status is unknown (no variant calls \
exist for this cell line). Do not omit this even when other trade-offs are \
more prominent. Then add any other genuine limitations.)
5. BEST FOR: (what experiment type suits this cell line best)"""

    return _chat(prompt)


def add_citations_to_justification(
    justification_text: str,
    dataset_citations: list[dict],
    literature: list[dict],
) -> str:
    """
    Append structured DATA SOURCES (section 6) and LITERATURE (section 7) blocks
    to LLM-generated justification text.

    Citations are built deterministically from evidence data rather than relying
    on the LLM to format them, which was unreliable.
    """
    citations_block = "\n\n6. DATA SOURCES:\n"
    for i, c in enumerate(dataset_citations, 1):
        citations_block += f"[D{i}] {c['name']}\n"
        citations_block += f"     {c['citation']}\n"
        citations_block += f"     PMID:{c['pmid']}\n"
        citations_block += f"     {c['url']}\n\n"

    citations_block += "\n7. LITERATURE:\n"
    for i, p in enumerate(literature, 1):
        citations_block += (
            f"[P{i}] {p['authors']} ({p['year']}). "
            f"{p['title']}.\n"
            f"      PMID:{p['pmid']} | {p['url']}\n\n"
        )

    return justification_text + citations_block


def generate_comparison(
    gene: str,
    top_results: list[dict],
    evidence_list: list[dict],
) -> str:
    """
    Generate a comparative summary across the top recommended cell lines.
    """
    lines = [f"TOP RECOMMENDATIONS FOR {gene}:"]
    for i, (res, ev) in enumerate(zip(top_results, evidence_list), 1):
        name    = res.get("official_name", res.get("cellosaurus_id", "?"))
        score   = res.get("scores", {}).get("final_score", 0)
        rna     = res.get("scores", {}).get("rna_score", 0)
        quality = res.get("scores", {}).get("quality_score", 0)
        disease = ev.get("metadata", {}).get("disease", "unknown")
        lines.append(
            f"  {i}. {name} — final={score:.2f}, RNA={rna:.2f}, "
            f"quality={quality:.2f}, disease={disease}"
        )

    summary_prompt = (
        "\n".join(lines)
        + f"\n\nIn 3-4 sentences, compare these options for studying {gene}. "
        "Highlight which offers the best expression evidence, which has the "
        "best data completeness, and whether any specific cell line stands out "
        "for a particular experimental context. Be direct and practical."
    )

    return _chat(summary_prompt)

