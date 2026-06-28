"""Agent 2 -- Verifier.

When the classifier flags a story as FAKE, this agent calls *another* service
(a web search) to surface what reputable sources actually report -- exactly the
assignment's example: "model finds fake news -> calls another app's API to
determine the real news". An optional LLM layer (Anthropic) summarizes the
findings when an API key is configured.
"""
import logging
from typing import List, Optional

from ..config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    VERIFY_MAX_RESULTS,
)
from ..schemas import Source, Verification

logger = logging.getLogger(__name__)


def _search(query: str, max_results: int) -> List[Source]:
    """Key-free web search via DuckDuckGo (ddgs)."""
    DDGS = None
    try:
        from ddgs import DDGS  # current package name
    except Exception:
        try:
            from duckduckgo_search import DDGS  # legacy package name
        except Exception as e:
            logger.warning("No search backend installed: %s", e)
            return []

    sources: List[Source] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                sources.append(
                    Source(
                        title=r.get("title", "") or "",
                        url=r.get("href", "") or r.get("url", "") or "",
                        snippet=r.get("body", "") or "",
                    )
                )
    except Exception as e:
        logger.warning("Search failed: %s", e)
    return sources


def _llm_summary(claim: str, sources: List[Source]) -> Optional[str]:
    if not ANTHROPIC_API_KEY:
        return None
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        context = "\n".join(f"- {s.title}: {s.snippet} ({s.url})" for s in sources)
        msg = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=320,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "A fake-news classifier flagged the claim below as likely FAKE. "
                        "Using ONLY the search results, explain in 3-4 factual sentences "
                        "what reputable sources actually say and whether the claim holds up.\n\n"
                        f"CLAIM:\n{claim[:1500]}\n\nSEARCH RESULTS:\n{context}"
                    ),
                }
            ],
        )
        return "".join(
            b.text for b in msg.content if getattr(b, "type", None) == "text"
        ).strip()
    except Exception as e:
        logger.warning("LLM summary failed: %s", e)
        return None


def verify_claim(claim: str, max_results: int = VERIFY_MAX_RESULTS) -> Verification:
    claim = (claim or "").strip()
    if not claim:
        return Verification(checked=False, method="none")

    query = claim if len(claim) <= 300 else claim[:300]
    sources = _search(query, max_results)
    if not sources:
        return Verification(
            checked=False,
            method="search",
            summary="No corroborating sources were found for this claim.",
        )

    summary = _llm_summary(claim, sources)
    method = "search+llm" if summary else "search"
    if not summary:
        summary = "Compare the claim against the sources below to assess its accuracy."
    return Verification(checked=True, method=method, summary=summary, sources=sources)
