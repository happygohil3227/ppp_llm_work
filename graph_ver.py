import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, TypedDict, Annotated
import operator

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from utils.vecDB_2 import LegalVectorDB

import torch
from transformers import AutoTokenizer, AutoModel
from openai import RateLimitError
import time


BASE_DIR = Path(__file__).resolve().parent
TEXT_DIR = BASE_DIR / "extracted_text"
VECTOR_DB_DIR = BASE_DIR / "vectorDB"
OUTPUT_DIR = BASE_DIR / "outputs"

TEXT_DIR.mkdir(exist_ok=True)
VECTOR_DB_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


SECTIONS: Dict[int, Dict[str, str]] = {
    1: {
        "name": "Context & Objective",
        "query": (
            "Extract the background, policy rationale, and purpose of this PPP "
            "concession agreement, including the problem it seeks to address, "
            "the public interest objective, the scope of the concession, and "
            "the intended outcomes for infrastructure service delivery and "
            "market efficiency"
        ),
    },
    2: {
        "name": "Scope of Concession & Rights Granted",
        "query": (
            "Extract clauses defining the scope of the concession, activities "
            "permitted to the concessionaire, rights granted to develop, "
            "operate, maintain, manage, or commercially exploit the project "
            "assets, and any explicit limitations or exclusions."
        ),
    },
    3: {
        "name": "Asset Ownership & Control",
        "query": (
            "Extract clauses specifying ownership of project assets, land, and "
            "equipment, rights of use versus title, and provisions allocating "
            "operational control, access, supervision, and decision-making authority."
        ),
    },
    4: {
        "name": "Regulatory & Operational Compliance",
        "query": (
            "Extract clauses imposing regulatory, operational, and compliance "
            "obligations, including approvals, directions, inspections, "
            "reporting requirements, standards, and the extent of supervisory "
            "or micromanagement authority over operations."
        ),
    },
    5: {
        "name": "Concession Period & Extension",
        "query": (
            "Extract clauses defining the concession term, commencement and "
            "expiry, conditions for extension or renewal, discretion of the "
            "authority, and any limits or uncertainty affecting the investment horizon."
        ),
    },
    6: {
        "name": "Tariff & Revenue Flexibility",
        "query": (
            "Extract all provisions governing how prices, tariffs, fees, or "
            "user charges are determined, revised, approved, or controlled; "
            "the extent of pricing discretion or regulation; revenue-sharing "
            "or levy mechanisms; indexation or adjustment formulas; and any "
            "restrictions affecting commercial revenue generation or innovation."
        ),
    },
    7: {
        "name": "Demand & Traffic Risk Allocation",
        "query": (
            "Extract clauses allocating demand, volume, traffic, or throughput "
            "risk, including minimum or assured traffic, exclusivity or "
            "non-exclusivity, diversion rights, competing facilities, "
            "capacity commitments, and any guarantees or disclaimers affecting traffic levels."
        ),
    },
    8: {
        "name": "Change in Law & Policy Risk",
        "query": (
            "Extract clauses addressing change in law, policy, regulation, or "
            "interpretation; allocation of resulting costs or benefits; "
            "compensation, relief, or adjustment mechanisms; and protections "
            "against adverse governmental or regulatory actions affecting the project."
        ),
    },
    9: {
        "name": "Relief Structure",
        "query": (
            "Extract clauses providing relief from contractual obligations due "
            "to force majeure, change in circumstances, or external shocks, "
            "including suspension, extension of time, cost or tariff adjustment, "
            "termination relief, and conditions or limitations on such relief."
        ),
    },
    10: {
        "name": "Termination & Step-in Rights",
        "query": (
            "Extract clauses governing termination events, defaults and cure "
            "periods, termination consequences, compensation or payout "
            "mechanisms, lender step-in or substitution rights, and protections "
            "preserving continuity of the project upon default or early termination."
        ),
    },
    11: {
        "name": "Dispute Resolution & Governing Law",
        "query": (
            "Extract clauses specifying dispute resolution mechanisms, "
            "escalation steps, arbitration or court processes, seat and venue, "
            "governing law, jurisdiction, timelines, enforceability of awards, "
            "and any provisions affecting neutrality or speed of dispute resolution."
        ),
    },
    12: {
        "name": "Assignment & Financing Flexibility",
        "query": (
            "Extract clauses governing assignment or transfer of rights and "
            "obligations, change of control, creation of security interests, "
            "financing or refinancing rights, lender protections, substitution "
            "or novation, and any approvals or restrictions affecting exit or refinancing."
        ),
    },
}


SECTION_PROMPT = ChatPromptTemplate.from_template(
    """
ROLE:
You are a senior infrastructure PPP legal–economic analyst advising
Government of India policy reform committees and multilateral lenders.

TASK:
Analyze the concession agreement ONLY for the section:
{section_name}

EVIDENCE DISCIPLINE (CRITICAL):
- Every legal conclusion MUST be backed by contract language
- Clearly distinguish between:
  (a) Explicit clauses
  (b) Implicit structure inferred from clauses
  (c) Absence or silence
- Do NOT speculate beyond the document
- If the document is silent, explicitly say so

OUTPUT FORMAT (STRICT — FOLLOW EXACTLY):

SECTION: {section_name}

LEGAL POSITION:
- What the contract explicitly allows, restricts, or mandates
- Clearly indicate if coverage is partial or conditional

ECONOMIC IMPLICATION:
- Commercial consequences strictly flowing from the contract text
- No assumptions beyond documented provisions

BANKABILITY IMPACT:
- Lender and investor perspective based on clarity, enforceability,
  and risk allocation in the contract

CLAUSE EVIDENCE:
- Verbatim excerpts from the document
- Include Article / Clause numbers wherever available
- Do NOT paraphrase in this section

EVIDENCE_STRENGTH: <STRONG | MODERATE | WEAK | NONE>
(one line only, uppercase, choose exactly one)

DOCUMENT CONTEXT:
{context}
"""
)


class AnalysisState(TypedDict):
    document_name: str
    text_path: Optional[Path]
    vecdb_path: Optional[Path]
    outputs_dir: Optional[Path]
    section_query_vectors: Dict[int, List[float]]
    section_results: Annotated[Dict[int, Dict], operator.or_]
    silence_flags: Annotated[List[str], operator.add]
    summary: Optional[str]


def _load_env() -> None:
    load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)


def _extract_confidence(result: str) -> str:
    match = re.search(
        r"EVIDENCE_STRENGTH:\\s*(STRONG|MODERATE|WEAK|NONE)",
        result.upper()
    )
    if not match:
        return "UNKNOWN"
    strength = match.group(1)
    if strength == "STRONG":
        return "HIGH"
    if strength == "MODERATE":
        return "MEDIUM"
    return "LOW"


def _section_confidence_score(result: str) -> str:
    quotes = len([ln for ln in result.splitlines() if ln.strip().startswith(("\"", "“"))])
    strength = _extract_confidence(result)

    if strength == "HIGH" and quotes >= 3:
        return "HIGH"
    if strength in {"HIGH", "MEDIUM"} and quotes >= 1:
        return "MEDIUM"
    if strength == "LOW" and quotes == 0:
        return "LOW"
    return "MEDIUM"


def _embed_section_queries() -> Dict[int, List[float]]:
    """
    Embed all section queries once (sequential) to avoid parallel model load.
    """
    model_name = "law-ai/InLegalBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name,
        use_safetensors=True,
        low_cpu_mem_usage=False
    )
    model.to("cpu")
    model.eval()

    vectors: Dict[int, List[float]] = {}
    with torch.no_grad():
        for sid, sec in SECTIONS.items():
            inputs = tokenizer(
                sec["query"],
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            outputs = model(**inputs)
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]
            summed = (token_embeddings * attention_mask.unsqueeze(-1)).sum(dim=1)
            counts = attention_mask.sum(dim=1)
            embedding = summed / counts
            vectors[sid] = embedding.squeeze().cpu().numpy().tolist()
    return vectors


def index_document(state: AnalysisState) -> AnalysisState:
    text_path = TEXT_DIR / f"{state['document_name']}.txt"
    if not text_path.exists():
        raise FileNotFoundError(f"Missing extracted text: {text_path}")

    vecdb_path = VECTOR_DB_DIR / state["document_name"]
    if not vecdb_path.exists():
        raise FileNotFoundError(f"Missing vector DB: {vecdb_path}")

    outputs_dir = OUTPUT_DIR / state["document_name"]
    outputs_dir.mkdir(parents=True, exist_ok=True)

    return {
        "document_name": state["document_name"],
        "text_path": text_path,
        "vecdb_path": vecdb_path,
        "outputs_dir": outputs_dir,
        "section_query_vectors": _embed_section_queries(),
        "section_results": {},
        "silence_flags": [],
        "summary": None,
    }


def analyze_section(state: AnalysisState, section_id: int) -> Dict:
    _load_env()
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY not set in .env")

    if not state.get("vecdb_path"):
        raise RuntimeError("Vector DB path not set before section analysis.")

    vecdb = LegalVectorDB.load(state["vecdb_path"])
    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=openai_key)
    chain = SECTION_PROMPT | llm | StrOutputParser()

    section = SECTIONS[section_id]
    query_vector = state["section_query_vectors"].get(section_id)
    if not query_vector:
        raise RuntimeError(f"Missing query embedding for section {section_id}")

    docs = vecdb.retrieve_for_section(
        query_vector=query_vector,
        top_k=12
    )
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "4"))
    base_wait = float(os.getenv("OPENAI_RETRY_BASE_SECONDS", "5"))

    for attempt in range(max_retries + 1):
        try:
            result = chain.invoke({
                "section_name": section["name"],
                "context": context
            })
            break
        except RateLimitError:
            if attempt >= max_retries:
                raise
            wait_s = base_wait * (2 ** attempt)
            time.sleep(wait_s)

    confidence = _extract_confidence(result)
    section_score = _section_confidence_score(result)

    section_json = {
        "document": state.document_name,
        "section_id": section_id,
        "section_name": section["name"],
        "query_used": section["query"],
        "analysis": result,
        "confidence": confidence,
        "section_confidence_score": section_score
    }

    if state.get("outputs_dir"):
        file_name = f"{section_id:02d}_{section['name'].replace(' ', '_')}.json"
        section_file = state["outputs_dir"] / file_name
        section_file.write_text(
            json.dumps(section_json, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    return {"section_results": {section_id: section_json}}


def silence_validator(state: AnalysisState) -> Dict:
    flags = []
    weak_sections = [
        s for s in state["section_results"].values()
        if s.get("confidence") in {"LOW", "UNKNOWN"}
    ]
    if len(weak_sections) >= 4:
        flags.append(
            "Multiple sections show LOW/UNKNOWN evidence strength. "
            "Re-check for false silence or missing clauses."
        )

    none_sections = [
        s for s in state["section_results"].values()
        if "EVIDENCE_STRENGTH: NONE" in s.get("analysis", "").upper()
    ]
    if len(none_sections) >= 2:
        flags.append(
            "Repeated EVIDENCE_STRENGTH: NONE across sections. "
            "Validate document completeness or ingestion quality."
        )

    return {"silence_flags": flags}


def assemble_report(state: AnalysisState) -> Dict:
    ordered = [state["section_results"][i] for i in sorted(state["section_results"].keys())]
    summary = {
        "document": state["document_name"],
        "sections": ordered,
        "silence_flags": state["silence_flags"]
    }
    return {"summary": json.dumps(summary, indent=2, ensure_ascii=False)}


def build_graph() -> StateGraph:
    graph = StateGraph(AnalysisState)

    graph.add_node("index_doc", index_document)

    for section_id in SECTIONS.keys():
        node_name = f"analyze_{section_id}"
        graph.add_node(
            node_name,
            lambda state, sid=section_id: analyze_section(state, sid)
        )
        graph.add_edge("index_doc", node_name)

    graph.add_node("validate_silence", silence_validator)
    graph.add_node("assemble_report", assemble_report)

    for section_id in SECTIONS.keys():
        graph.add_edge(f"analyze_{section_id}", "validate_silence")

    graph.add_edge("validate_silence", "assemble_report")
    graph.add_edge("assemble_report", END)

    graph.set_entry_point("index_doc")
    return graph


def run(document_name: str, max_concurrency: int = 2) -> AnalysisState:
    graph = build_graph().compile()
    state = {"document_name": document_name}
    return graph.invoke(state, config={"max_concurrency": max_concurrency})


if __name__ == "__main__":
    # Example usage:
    # python graph_ver.py "Bahuli_IOCL_compressed"
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python graph_ver.py <DOCUMENT_NAME>")

    max_concurrency = int(os.getenv("OPENAI_MAX_CONCURRENCY", "2"))
    final_state = run(sys.argv[1], max_concurrency=max_concurrency)
    print(final_state.summary)
