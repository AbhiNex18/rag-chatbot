import os
import sys
import time
import re
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

from dotenv import load_dotenv

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

if __package__:
    from .retriever import KBRetriever
    from .tools import check_order_status, extract_order_id
else:
    from agent.retriever import KBRetriever
    from agent.tools import check_order_status, extract_order_id

LOG_FILE = os.path.join(_PROJECT_ROOT, "agent_reasoning.log")
logger = logging.getLogger("support_agent")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

ESCALATE_PHRASE = "I don't know, escalating to a human agent."

# Policy / FAQ language — KB even if the query mentions "order" or "shipping".
KB_INTENT_PATTERNS = [
    r"\bfree shipping\b",
    r"\bshipping (policy|fee|cost|price|rate|options?|methods?|timelines?)\b",
    r"\bhow long does (?:standard )?shipping take\b",
    r"\bdo you (offer|provide|have|accept)\b",
    r"\breturn policy\b",
    r"\bcan i return\b",
    r"\bhow (?:do i|to) return\b",
    r"\breturn(?:ing)? (?:my |the )?(?:product|item|order|purchase)\b",
    r"\bsend (?:it |them )?back\b",
    r"\brefund\b",
    r"\bwarranty\b",
    r"\bsizing\b|\bsize chart\b",
    r"\bpayment methods?\b",
    r"\bgift cards?\b",
    r"\bpromo code\b|\bdiscount\b",
    r"\bprice match",
    r"\bcancel(?:lation)? (?:my |an |the )?order\b",
    r"\bcan i cancel\b",
    r"\bpre-?order",
    r"\bwholesale\b|\bbulk\b",
    r"\bprivacy\b|\bpassword\b|\blogin\b",
    r"\bcustoms\b|\bduties\b",
    r"\bsubscription\b",
    r"\borders over\b",
    r"\bwhat is your (return|shipping|warranty|privacy)",
]

# Genuine package/order tracking — not store policy.
TOOL_INTENT_PATTERNS = [
    r"\btrack(?:ing)?(?:\s+my)?(?:\s+order|\s+package)?\b",
    r"\border status\b",
    r"\bstatus of (?:my |the )?order\b",
    r"\bwhere(?:'s| is) (?:my |the )?(?:order|package)\b",
    r"\bdelivery status\b",
    r"\bcheck (?:on |if )?(?:my )?(?:order|package)\b",
    r"\bcheck (?:the )?status\b",
    r"\bwhen will (?:my |the )?(?:order|package) arrive\b",
    r"\bhas (?:my |the )?order shipped\b",
    r"\border has shipped\b",
    r"\bif (?:my |the )?order has shipped\b",
    r"\blook up (?:my )?order\b",
    r"\bwhat items are included in order\b",
    r"\bwhere is package\b",
]

OUT_OF_SCOPE_PATTERNS = [
    r"\bteleportation\b",
    r"\bquantum\b",
    r"\brecipe\b",
    r"\bpizza\b",
    r"\bpython\b|\bpytorch\b|\bwrite (?:a )?code\b|\bscript\b",
    r"\bpresident\b",
    r"\bcapital of\b",
    r"\bastronomy\b",
    r"\bsolve equation\b",
    r"\bmeaning of life\b",
    r"\balien\b",
    r"\btime travel\b",
    r"\bsuperpower\b",
    r"\bbitcoin\b|\bcrypto\b|\bstock market\b",
    r"\bworld cup\b",
    r"\bhyper-travel\b",
]

AMBIGUOUS_KB_WORDS = {
    "returns", "shipping", "sizing", "warranty", "refunds", "payments", "tracking",
}
INTENT_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "it", "this", "that",
    "me", "my", "i", "we", "you", "your", "with", "please", "can", "could",
    "would", "should", "to", "for", "of", "and", "or", "on", "in", "at",
    "help", "stuff", "something", "thing", "issue", "problem", "info",
    "details", "good", "about",
}
CANCEL_CLARIFICATION_RE = re.compile(
    r"\b(never mind|nevermind|cancel that|forget it|stop|new question)\b", re.IGNORECASE
)
STANDALONE_ID_RE = re.compile(r"^\s*(?:ORD-?)?\d{4,6}\s*$", re.IGNORECASE)
ID_FOLLOWUP_RE = re.compile(
    r"^\s*(?:(?:it(?:'s| is)|this is|here(?:'s| is)|my|the)?\s*)?"
    r"(?:ord(?:er)?|order)\s*(?:id|number|no\.?)?\s*(?:is|:)?\s*"
    r"(?:ORD-?)?\d{4,6}\s*$",
    re.IGNORECASE,
)


def _redact_secrets(text: str) -> str:
    text = re.sub(r"sk-[A-Za-z0-9_\-]+", "sk-***", text)
    text = re.sub(r"hf_[A-Za-z0-9]+", "hf_***", text)
    return text


def _is_id_followup(query: str) -> bool:
    q = (query or "").strip()
    return bool(STANDALONE_ID_RE.fullmatch(q) or ID_FOLLOWUP_RE.fullmatch(q))


class SupportAgent:
    """
    Agentic Customer Support Bot.

    Conversation flags live in the per-session dict passed to process_query —
    never on this cached instance, so Streamlit users cannot leak context.
    """

    SIMILARITY_THRESHOLD = 0.52

    def __init__(self, kb_dir: Optional[str] = None):
        self.retriever = KBRetriever(kb_dir=kb_dir)
        self.retriever.load_or_build_index()
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        self.anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    def _classify_intent(self, query: str, session: Optional[Dict[str, Any]] = None) -> str:
        session = session or {}
        awaiting = bool(session.get("awaiting_order_id"))
        q_lower = query.lower().strip()
        order_id = extract_order_id(query, allow_bare_number=awaiting)
        only_id = bool(STANDALONE_ID_RE.fullmatch(query.strip()))

        if awaiting and CANCEL_CLARIFICATION_RE.search(query):
            return "ambiguous"

        if awaiting and (only_id or order_id):
            return "tool_needed"

        if any(re.search(p, q_lower) for p in OUT_OF_SCOPE_PATTERNS):
            return "out_of_scope"

        kb_hit = any(re.search(p, q_lower) for p in KB_INTENT_PATTERNS)
        tool_hit = any(re.search(p, q_lower) for p in TOOL_INTENT_PATTERNS)

        if kb_hit and not tool_hit:
            return "kb_question"
        if tool_hit:
            return "tool_needed"

        id_followup = _is_id_followup(query)
        last_intent = session.get("last_intent")
        if (only_id or id_followup) and order_id:
            if last_intent == "tool_needed" or awaiting:
                return "tool_needed"
            if last_intent == "kb_question":
                return "kb_question"
            return "ambiguous"

        if order_id and not kb_hit:
            return "tool_needed"

        tokens = q_lower.split()
        if len(tokens) <= 2 and q_lower not in AMBIGUOUS_KB_WORDS:
            return "ambiguous"

        content_tokens = [
            t for t in re.findall(r"[a-z0-9]+", q_lower)
            if t not in INTENT_STOPWORDS and len(t) > 1
        ]
        if len(content_tokens) <= 1 and q_lower not in AMBIGUOUS_KB_WORDS:
            return "ambiguous"

        return "kb_question"

    def _generate_llm_response(
        self, query: str, context: str, docs: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        meta = {
            "llm_provider": None,
            "llm_status": "unused",
            "fallback_used": False,
            "error": None,
        }

        if not self.openai_key:
            meta["llm_status"] = "no_api_key"
            meta["fallback_used"] = True
            logger.info("OpenAI skipped: OPENAI_API_KEY is not set. Using grounded fallback.")
            return self._template_fallback(docs), meta

        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.openai_key)
            system = (
                "You are a careful customer support agent. Answer ONLY using the "
                "Knowledge Base context. If the context does not contain the answer, "
                f"reply with exactly: {ESCALATE_PHRASE}\n"
                "Do not invent policies, prices, dates, or tracking numbers. "
                "Cite the policy source filename when you answer."
            )
            messages = [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": f"Knowledge Base context:\n{context}\n\nCustomer question: {query}",
                },
            ]
            last_error = None
            for attempt in range(2):
                try:
                    res = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages,
                        temperature=0.0,
                    )
                    answer = (res.choices[0].message.content or "").strip()
                    meta["llm_provider"] = "openai"
                    meta["llm_status"] = "success"
                    logger.info("OpenAI response generated successfully (gpt-4o-mini).")
                    return answer, meta
                except Exception as inner:
                    last_error = inner
                    err_l = str(inner).lower()
                    if attempt == 0 and ("rate limit" in err_l or "429" in err_l):
                        logger.warning("OpenAI rate-limited; retrying once.")
                        time.sleep(1.5)
                        continue
                    raise
            if last_error:
                raise last_error
        except Exception as e:
            err_name = e.__class__.__name__
            err_text = _redact_secrets(str(e))
            meta["error"] = f"{err_name}: {err_text}"
            meta["fallback_used"] = True
            lowered = err_text.lower()
            if "authentication" in lowered or "invalid api key" in lowered or "incorrect api" in lowered:
                status = "invalid_api_key"
            elif "rate limit" in lowered or "429" in lowered:
                status = "rate_limited"
            elif "insufficient_quota" in lowered or "quota" in lowered:
                status = "quota_exceeded"
            elif "timeout" in lowered or "connection" in lowered:
                status = "network_error"
            else:
                status = "error"
            meta["llm_status"] = status
            logger.warning("OpenAI call failed (%s). Using grounded fallback. %s", status, err_text)
            return self._template_fallback(docs), meta

    @staticmethod
    def _clean_chunk_text(content: str) -> str:
        """Strip embedding prefixes like 'Source: file.md | Topic: ...' from user-facing text."""
        if not content:
            return ""
        lines = content.split("\n", 1)
        if lines[0].startswith("Source:"):
            content = lines[1] if len(lines) > 1 else ""
        return content.strip()

    def _template_fallback(self, docs: Optional[List[Dict[str, Any]]]) -> str:
        if not docs:
            return (
                "I could not generate an LLM answer, and no policy excerpt was available. "
                "I can escalate this to a human agent."
            )

        top_source = docs[0].get("source") or "policy"
        bodies = []
        for doc in docs:
            if (doc.get("source") or "policy") != top_source:
                continue
            body = self._clean_chunk_text(doc.get("content", ""))
            if body and body not in bodies:
                bodies.append(body)
            if len(bodies) >= 3:
                break

        excerpt = f"**{top_source}**\n\n" + "\n\n".join(bodies)
        return (
            f"{excerpt}\n\n"
            "_Quoted from the knowledge base because OpenAI was unavailable "
            "(the full LLM answer was not generated)._"
        )

    def _looks_like_unknown(self, answer: str) -> bool:
        lowered = answer.lower()
        return (
            ESCALATE_PHRASE.lower() in lowered
            or "escalating to a human" in lowered
            or lowered.startswith("i don't know")
            or lowered.startswith("i do not know")
        )

    def _result(
        self,
        *,
        query: str,
        intent: str,
        decision: str,
        confidence_score: float,
        response: str,
        reasoning: str,
        latency_ms: float,
        retrieved_docs: Optional[List] = None,
        tool_call: Optional[Dict] = None,
        llm_meta: Optional[Dict] = None,
        session: Optional[Dict[str, Any]] = None,
        awaiting_order_id: bool = False,
    ) -> Dict[str, Any]:
        if session is not None:
            session["awaiting_order_id"] = awaiting_order_id
            session["last_intent"] = intent
            session["last_decision"] = decision
            if query and not _is_id_followup(query):
                session["last_user_query"] = query

        llm_meta = llm_meta or {"llm_status": "unused", "fallback_used": False, "error": None}
        sources = [d.get("source") for d in (retrieved_docs or []) if d.get("source")]
        res = {
            "query": query,
            "intent": intent,
            "decision": decision,
            "confidence_score": confidence_score,
            "retrieved_docs": retrieved_docs or [],
            "tool_call": tool_call,
            "response": response,
            "reasoning": reasoning,
            "latency_ms": latency_ms,
            "llm_status": llm_meta.get("llm_status"),
            "fallback_used": bool(llm_meta.get("fallback_used")),
            "llm_error": llm_meta.get("error"),
            "sources": sources,
        }
        self._log_decision(res)
        return res

    def process_query(
        self, query: str, session: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        start_time = time.time()
        if session is None:
            session = {}
        query_str = query.strip() if query else ""

        def latency() -> float:
            return round((time.time() - start_time) * 1000, 2)

        if not query_str:
            return self._result(
                query=query or "",
                intent="ambiguous",
                decision="ASK_CLARIFICATION",
                confidence_score=0.0,
                response="Please enter a valid question or order ID.",
                reasoning="Empty query received.",
                latency_ms=0.0,
                session=session,
                awaiting_order_id=bool(session.get("awaiting_order_id")),
            )

        awaiting = bool(session.get("awaiting_order_id"))
        if awaiting and CANCEL_CLARIFICATION_RE.search(query_str):
            return self._result(
                query=query_str,
                intent="ambiguous",
                decision="ASK_CLARIFICATION",
                confidence_score=0.0,
                response=(
                    "No problem — I cancelled the order lookup. "
                    "How else can I help with returns, shipping, sizing, or tracking?"
                ),
                reasoning="User cancelled pending order-ID clarification.",
                latency_ms=latency(),
                session=session,
                awaiting_order_id=False,
            )

        intent = self._classify_intent(query_str, session)
        extracted_id = extract_order_id(query_str, allow_bare_number=awaiting)

        if intent == "tool_needed":
            if not extracted_id:
                return self._result(
                    query=query_str,
                    intent=intent,
                    decision="ASK_CLARIFICATION",
                    confidence_score=1.0,
                    response=(
                        "I can check your order status. Please reply with your Order ID "
                        "(for example, ORD-1001)."
                    ),
                    reasoning=(
                        f"Intent=tool_needed for '{query_str}' but no order ID was present. "
                        "Do not guess parameters; request the Order ID."
                    ),
                    latency_ms=latency(),
                    tool_call={"tool": "check_order_status", "status": "MISSING_PARAM"},
                    session=session,
                    awaiting_order_id=True,
                )

            tool_result = check_order_status(extracted_id)
            if tool_result.get("found"):
                status = tool_result["status"]
                items_str = ", ".join(
                    [f"{item['quantity']}x {item['name']}" for item in tool_result["items"]]
                )
                response_text = (
                    f"Order **{tool_result['order_id']}** is currently **{status.upper()}**.\n"
                    f"- **Estimated Delivery**: {tool_result['eta']}\n"
                    f"- **Items**: {items_str}\n"
                    f"- **Tracking Number**: {tool_result['tracking_number']}"
                )
                reasoning = (
                    f"Executed check_order_status({extracted_id}); found status={status}. "
                    "Pending order-ID clarification cleared."
                )
            else:
                response_text = (
                    f"I looked up **{extracted_id}** in our order system and did not find a match. "
                    "Please double-check the Order ID (format ORD-1001) and try again."
                )
                reasoning = (
                    f"Executed check_order_status({extracted_id}); order not in database. "
                    "Did not fabricate tracking details."
                )

            return self._result(
                query=query_str,
                intent=intent,
                decision="CALL_TOOL",
                confidence_score=1.0,
                response=response_text,
                reasoning=reasoning,
                latency_ms=latency(),
                tool_call={
                    "tool": "check_order_status",
                    "order_id": extracted_id,
                    "result": tool_result,
                },
                session=session,
                awaiting_order_id=False,
            )

        if intent == "out_of_scope":
            return self._result(
                query=query_str,
                intent=intent,
                decision="ESCALATE",
                confidence_score=0.0,
                response=f"{ESCALATE_PHRASE} (Reason: this question is outside customer support scope.)",
                reasoning="Query classified out-of-scope. No KB retrieval or tool call.",
                latency_ms=latency(),
                session=session,
                awaiting_order_id=False,
            )

        if intent == "ambiguous":
            return self._result(
                query=query_str,
                intent=intent,
                decision="ASK_CLARIFICATION",
                confidence_score=0.0,
                response=(
                    "Could you please rephrase that? I can help with returns, shipping, "
                    "sizing, warranties, payments, or looking up a specific Order ID."
                ),
                reasoning=f"Query '{query_str}' is too vague to route.",
                latency_ms=latency(),
                session=session,
                awaiting_order_id=False,
            )

        retrieve_query = query_str
        continued_id = None
        if _is_id_followup(query_str) and session.get("last_user_query"):
            continued_id = extract_order_id(query_str, allow_bare_number=True)
            retrieve_query = session["last_user_query"]

        docs, confidence_score = self.retriever.search_with_confidence(retrieve_query, top_k=3)
        top_source = docs[0]["source"] if docs else "none"
        overlap = docs[0].get("term_overlap", []) if docs else []

        if confidence_score < self.SIMILARITY_THRESHOLD:
            return self._result(
                query=query_str,
                intent=intent,
                decision="ESCALATE",
                confidence_score=confidence_score,
                response=ESCALATE_PHRASE,
                reasoning=(
                    f"Top retrieval score {confidence_score:.4f} < threshold "
                    f"{self.SIMILARITY_THRESHOLD} (best source={top_source}). "
                    "Insufficient evidence; escalating rather than guessing."
                ),
                latency_ms=latency(),
                retrieved_docs=docs,
                session=session,
                awaiting_order_id=False,
            )

        context_block = "\n---\n".join(
            [f"[{d['source']}]: {self._clean_chunk_text(d['content'])}" for d in docs]
        )
        answer, llm_meta = self._generate_llm_response(query_str, context_block, docs)

        if self._looks_like_unknown(answer):
            return self._result(
                query=query_str,
                intent=intent,
                decision="ESCALATE",
                confidence_score=confidence_score,
                response=ESCALATE_PHRASE,
                reasoning=(
                    f"Retrieval score {confidence_score:.4f} (source={top_source}, overlap={overlap}) "
                    "but the grounded answer could not be supported by context. "
                    f"llm_status={llm_meta.get('llm_status')} fallback={llm_meta.get('fallback_used')}."
                ),
                latency_ms=latency(),
                retrieved_docs=docs,
                llm_meta=llm_meta,
                session=session,
                awaiting_order_id=False,
            )

        gen_note = (
            "OpenAI gpt-4o-mini succeeded"
            if llm_meta.get("llm_status") == "success"
            else (
                f"OpenAI not used (status={llm_meta.get('llm_status')}; "
                f"fallback={llm_meta.get('fallback_used')})"
            )
        )
        reasoning = (
            f"Retrieval score {confidence_score:.4f} >= {self.SIMILARITY_THRESHOLD}. "
            f"Top source={top_source}; overlap={overlap}. {gen_note}."
        )
        if llm_meta.get("error"):
            reasoning += f" LLM error: {_redact_secrets(str(llm_meta['error']))}"

        return self._result(
            query=query_str,
            intent=intent,
            decision="ANSWER_KB",
            confidence_score=confidence_score,
            response=answer,
            reasoning=reasoning,
            latency_ms=latency(),
            retrieved_docs=docs,
            llm_meta=llm_meta,
            session=session,
            awaiting_order_id=False,
        )

    def _log_decision(self, result: Dict[str, Any]):
        sources = result.get("sources") or [
            d.get("source") for d in result.get("retrieved_docs") or []
        ]
        tool = result.get("tool_call") or {}
        log_entry = (
            f"\n--- DECISION LOG [{datetime.now().isoformat()}] ---\n"
            f"Query: {result['query']}\n"
            f"Intent: {result['intent']} | Decision: {result['decision']} | "
            f"Confidence: {result['confidence_score']:.4f}\n"
            f"Sources: {sources}\n"
            f"Tool: {tool.get('tool')} status={tool.get('status')} order_id={tool.get('order_id')}\n"
            f"LLM status: {result.get('llm_status')} | fallback_used: {result.get('fallback_used')}\n"
            f"LLM error: {_redact_secrets(str(result.get('llm_error')))}\n"
            f"Latency: {result['latency_ms']} ms\n"
            f"Reasoning: {result['reasoning']}\n"
            f"Response: {result['response']}\n"
            f"--------------------------------------------------\n"
        )
        logger.info(_redact_secrets(log_entry))


if __name__ == "__main__":
    agent = SupportAgent()
    session: Dict[str, Any] = {}
    sample_queries = [
        "What is your return policy window?",
        "Do you offer free shipping on orders over $50?",
        "Can you check my order status for ORD-1002?",
        "Where is my package?",
        "ORD-1002",
        "What is your policy on quantum teleportation shipping?",
        "Who is the president of the USA?",
    ]
    for q in sample_queries:
        print(f"\nQUERY: {q}")
        res = agent.process_query(q, session)
        print(
            f"INTENT: {res['intent']} | DECISION: {res['decision']} | "
            f"Score: {res['confidence_score']:.2f}"
        )
        print(f"RESPONSE: {res['response'][:240]}")
