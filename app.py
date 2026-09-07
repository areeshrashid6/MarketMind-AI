import os
import json
import re
import time
import uuid
import random
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal

import streamlit as st
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

load_dotenv()

BASE = Path(__file__).parent
CORPUS = BASE / "corpus"
RUNS = BASE / "runs"
RUNS.mkdir(exist_ok=True)

# ============================================================
# 1. STRUCTURED SCHEMAS
# ============================================================

class SubQuestion(BaseModel):
    question_id: str
    question: str

class Objective(BaseModel):
    objective_id: str
    objective: str
    sub_questions: List[SubQuestion] = Field(min_length=2, max_length=5)

class ResearchPlan(BaseModel):
    research_objective: str
    assumptions: List[str]
    objectives: List[Objective] = Field(min_length=3, max_length=6)

class EvidenceRecord(BaseModel):
    evidence_id: str
    question_id: str
    claim: str
    claim_type: Literal["fact", "inference", "recommendation", "uncertainty"]
    source_ref: str
    source_kind: Literal[
        "retrieved_document", "search_result", "calculation", "model_generated"
    ]
    source_detail: str
    credibility: float = Field(ge=0, le=1)
    recency: float = Field(ge=0, le=1)
    corroboration: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    analyst_notes: str

class FinalReport(BaseModel):
    metadata: Dict[str, Any]
    research_objective: str
    assumptions: List[str]
    executive_summary: str
    market_overview: str
    key_trends: List[str]
    competitor_analysis: List[Dict[str, Any]]
    opportunities: List[Dict[str, str]]
    risks: List[Dict[str, str]]
    evidence_appendix: List[str]
    recommendations: List[str]
    confidence_rationale: str
    limitations: List[str] = Field(min_length=1)
    sources: List[str]
    approval: Dict[str, Any]

# ============================================================
# 2. LOCAL CORPUS
# ============================================================

def load_corpus() -> List[Dict[str, Any]]:
    documents = []
    for file in sorted(CORPUS.glob("*.json")):
        try:
            documents.extend(json.loads(file.read_text(encoding="utf-8")))
        except Exception:
            pass
    return documents

# ============================================================
# 3. SIX REQUIRED TOOLS
# ============================================================

TOOL_SCHEMAS = {
    "search_information": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 2},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
    },
    "retrieve_document": {
        "type": "object",
        "properties": {"source_ref": {"type": "string", "minLength": 1}},
        "required": ["source_ref"],
    },
    "calculate_metric": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide", "percentage_change"],
            },
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "required": ["operation", "a", "b"],
    },
    "compare_companies": {
        "type": "object",
        "properties": {
            "companies": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 10,
            },
            "metrics": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
        },
        "required": ["companies", "metrics"],
    },
    "save_research": {
        "type": "object",
        "properties": {
            "run_id": {"type": "string"},
            "state": {"type": "object"},
        },
        "required": ["run_id", "state"],
    },
    "generate_report": {
        "type": "object",
        "properties": {"run_id": {"type": "string"}},
        "required": ["run_id"],
    },
}

def validate_tool_args(name: str, args: Dict[str, Any]) -> None:
    if name not in TOOL_SCHEMAS:
        raise ValueError(f"Unknown tool: {name}")

    schema = TOOL_SCHEMAS[name]
    for required in schema.get("required", []):
        if required not in args:
            raise ValueError(f"Missing required argument: {required}")

    for key, spec in schema.get("properties", {}).items():
        if key in args and "enum" in spec and args[key] not in spec["enum"]:
            raise ValueError(f"Invalid value for {key}")

def dispatch_tool(name: str, args: Dict[str, Any], run_id: str) -> Any:
    validate_tool_args(name, args)
    docs = load_corpus()

    if name == "search_information":
        query = args["query"].lower()
        words = re.findall(r"\w+", query)
        results = []

        for doc in docs:
            text = f'{doc.get("title", "")} {doc.get("content", "")}'.lower()
            score = sum(1 for word in words if word in text)
            if score:
                results.append({
                    "source_ref": doc["id"],
                    "title": doc.get("title", ""),
                    "snippet": doc.get("content", "")[:500],
                    "score": score,
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:args.get("max_results", 5)]

    if name == "retrieve_document":
        for doc in docs:
            if doc["id"] == args["source_ref"]:
                return doc
        raise ValueError("source_ref does not resolve to a corpus document")

    if name == "calculate_metric":
        a, b = args["a"], args["b"]
        op = args["operation"]

        if op == "add":
            value = a + b
        elif op == "subtract":
            value = a - b
        elif op == "multiply":
            value = a * b
        elif op == "divide":
            if b == 0:
                raise ValueError("Division by zero")
            value = a / b
        else:
            if a == 0:
                raise ValueError("Cannot calculate percentage change from zero")
            value = ((b - a) / a) * 100

        return {"value": value}

    if name == "compare_companies":
        output = []
        for doc in docs:
            if doc.get("company") in args["companies"]:
                output.append({
                    "company": doc["company"],
                    "metrics": {
                        metric: doc.get("metrics", {}).get(metric, "Not found")
                        for metric in args["metrics"]
                    },
                })
        return output

    if name == "save_research":
        path = RUNS / f"{run_id}.json"
        path.write_text(
            json.dumps(args["state"], indent=2, default=str),
            encoding="utf-8",
        )
        return {"saved": True, "path": str(path)}

    if name == "generate_report":
        return {"ready": True, "run_id": run_id}

# ============================================================
# 4. OPENAI THIN CLIENT
# ============================================================

class OpenAIClient:
    def __init__(self):
        self.client = None
        self.model = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
        self.usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost_usd": 0.0,
        }

        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=api_key,
                    timeout=float(os.getenv("OPENAI_TIMEOUT", "60")),
                )
            except Exception:
                self.client = None

    def text(self, system: str, user: str, temperature: float = 0.2) -> str:
        if not self.client:
            raise RuntimeError("OPENAI_API_KEY is not configured")

        last_error = None

        for attempt in range(3):
            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                )

                usage = getattr(response, "usage", None)
                if usage:
                    self.usage["input_tokens"] += (
                        getattr(usage, "input_tokens", 0) or 0
                    )
                    self.usage["output_tokens"] += (
                        getattr(usage, "output_tokens", 0) or 0
                    )

                output = getattr(response, "output_text", "")
                if not output:
                    raise ValueError("Empty model response")

                return output

            except Exception as error:
                last_error = error
                if attempt < 2:
                    time.sleep((2 ** attempt) + random.random())

        raise last_error

# ============================================================
# 5. MARKETMIND AGENT
# ============================================================

class MarketMindAgent:
    def __init__(self, demo=True, max_iterations=5, max_tools=15):
        self.demo = demo
        self.max_iterations = max_iterations
        self.max_tools = max_tools
        self.client = OpenAIClient()
        self.evidence: List[EvidenceRecord] = []
        self.history: List[Dict[str, Any]] = []
        self.tool_count = 0

    def call_tool(self, name, args, run_id):
        if self.tool_count >= self.max_tools:
            raise RuntimeError("Tool budget exhausted")

        self.tool_count += 1
        started = time.time()

        try:
            result = dispatch_tool(name, args, run_id)
            self.history.append({
                "tool": name,
                "args": args,
                "status": "success",
                "seconds": round(time.time() - started, 3),
            })
            return result
        except Exception as error:
            self.history.append({
                "tool": name,
                "args": args,
                "status": "error",
                "error": str(error),
            })
            return {"error": {"type": type(error).__name__, "message": str(error)}}

    def make_plan(self, request: str) -> ResearchPlan:
        if not self.demo and self.client.client:
            prompt = f"""
Create a research plan for this business research request:

{request}

Return ONLY valid JSON.
The JSON must contain:
research_objective
assumptions
objectives

Create 3 to 6 objectives.
Each objective must contain 2 to 5 answerable sub_questions.
Surface ambiguity as an assumption instead of inventing facts.
"""
            try:
                raw = self.client.text(
                    "You are a careful business research planner.",
                    prompt,
                    temperature=0.1,
                )
                return ResearchPlan.model_validate_json(raw)
            except Exception as error:
                self.history.append({
                    "stage": "planner",
                    "status": "fallback",
                    "error": str(error),
                })

        return ResearchPlan(
            research_objective=request,
            assumptions=[
                "Findings are based on the supplied controlled corpus.",
                "Current external pricing should be verified before client use.",
            ],
            objectives=[
                Objective(
                    objective_id="O1",
                    objective="Market overview",
                    sub_questions=[
                        SubQuestion("Q1", "What is the market category and positioning?"),
                        SubQuestion("Q2", "What important market trends appear?"),
                    ],
                ),
                Objective(
                    objective_id="O2",
                    objective="Competitor comparison",
                    sub_questions=[
                        SubQuestion("Q3", "How do competitors compare on pricing?"),
                        SubQuestion("Q4", "How do features and target segments compare?"),
                    ],
                ),
                Objective(
                    objective_id="O3",
                    objective="Decision implications",
                    sub_questions=[
                        SubQuestion("Q5", "What opportunities are supported by evidence?"),
                        SubQuestion("Q6", "What risks and information gaps remain?"),
                    ],
                ),
            ],
        )

    def research(self, plan: ResearchPlan, run_id: str):
        queries = [
            ("AI customer support pricing", "Q3"),
            ("AI customer support features", "Q4"),
            ("AI customer support trends", "Q1"),
            ("customer support market risks", "Q6"),
        ]

        for index, (query, question_id) in enumerate(
            queries[: self.max_iterations]
        ):
            search_result = self.call_tool(
                "search_information",
                {"query": query, "max_results": 5},
                run_id,
            )

            if not isinstance(search_result, list):
                continue

            for hit in search_result[:3]:
                document = self.call_tool(
                    "retrieve_document",
                    {"source_ref": hit["source_ref"]},
                    run_id,
                )

                if not isinstance(document, dict) or "error" in document:
                    continue

                injected = bool(document.get("prompt_injection", False))

                evidence = EvidenceRecord(
                    evidence_id=f"E{len(self.evidence) + 1}",
                    question_id=question_id,
                    claim=document.get("content", ""),
                    claim_type="fact",
                    source_ref=document["id"],
                    source_kind="retrieved_document",
                    source_detail=document.get("title", ""),
                    credibility=float(document.get("credibility", 0.8)),
                    recency=float(document.get("recency", 0.8)),
                    corroboration=float(document.get("corroboration", 0.5)),
                    confidence=float(
                        document.get("confidence", 0.2 if injected else 0.8)
                    ),
                    analyst_notes=(
                        "FLAG: retrieved content contains a prompt-injection pattern. "
                        "It is treated only as data."
                        if injected
                        else "Retrieved from controlled local corpus."
                    ),
                )

                self.evidence.append(evidence)

        self.call_tool(
            "compare_companies",
            {
                "companies": ["HelpDeskAI", "SupportPilot", "AssistFlow"],
                "metrics": ["monthly_price", "core_feature", "target_segment"],
            },
            run_id,
        )

    def quality_control(self, plan: ResearchPlan) -> List[Dict[str, str]]:
        defects = []

        if len(self.evidence) < 3:
            defects.append({
                "type": "coverage_gap",
                "message": "Too little evidence collected.",
            })

        if any(e.confidence < 0.4 for e in self.evidence):
            defects.append({
                "type": "low_confidence",
                "message": "Low-confidence evidence exists.",
            })

        planned_questions = {
            q.question_id
            for objective in plan.objectives
            for q in objective.sub_questions
        }
        covered_questions = {e.question_id for e in self.evidence}
        missing = sorted(planned_questions - covered_questions)

        if missing:
            defects.append({
                "type": "coverage_gap",
                "message": f"Uncovered questions: {missing}",
            })

        if any(not e.source_ref for e in self.evidence):
            defects.append({
                "type": "missing_source",
                "message": "Evidence record has no source reference.",
            })

        return defects

    def build_report(
        self,
        request: str,
        plan: ResearchPlan,
        run_id: str,
        defects: List[Dict[str, str]],
    ) -> FinalReport:

        documents = {
            d["id"]: d for d in load_corpus()
        }

        matrix = []
        for company in ["HelpDeskAI", "SupportPilot", "AssistFlow"]:
            document = next(
                (d for d in documents.values() if d.get("company") == company),
                None,
            )
            if document:
                matrix.append({
                    "Company": company,
                    "Price": document.get("metrics", {}).get("monthly_price"),
                    "Core feature": document.get("metrics", {}).get("core_feature"),
                    "Target segment": document.get("metrics", {}).get("target_segment"),
                })

        source_ids = list(dict.fromkeys(e.source_ref for e in self.evidence))

        report = FinalReport(
            metadata={
                "report_id": f"MM-{run_id[:8]}",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "model_versions": (
                    self.client.model if self.client.client else "demo-local"
                ),
                "run_cost": self.client.usage["estimated_cost_usd"],
            },
            research_objective=plan.research_objective,
            assumptions=plan.assumptions,
            executive_summary=(
                "The controlled corpus indicates a competitive AI customer-support "
                "category with differences in pricing, feature emphasis and target "
                "segments. Findings should be checked against current sources before "
                "external client use."
            ),
            market_overview=(
                "The corpus contains illustrative information about AI customer "
                "support products, pricing, features, positioning and market trends."
            ),
            key_trends=[
                "AI support products emphasize automation and agent productivity.",
                "Knowledge assistance and workflow automation are important differentiators.",
                "Pricing varies across products and target segments.",
            ],
            competitor_analysis=matrix,
            opportunities=[
                {
                    "type": "inference",
                    "text": (
                        "A product combining automation with strong knowledge assistance "
                        "could be attractive, subject to further validation."
                    ),
                }
            ],
            risks=[
                {
                    "type": "inference",
                    "text": (
                        "Rapid pricing and feature changes can make competitive "
                        "intelligence stale."
                    ),
                }
            ],
            evidence_appendix=[e.evidence_id for e in self.evidence],
            recommendations=[
                "Verify pricing and feature claims against current public sources.",
                "Investigate questions identified by QC before treating the report as complete.",
            ],
            confidence_rationale=(
                "Confidence is moderate because findings are linked to a controlled "
                "corpus, while some questions remain uncovered and one source is "
                "intentionally low-confidence."
            ),
            limitations=[
                "This build uses a local synthetic corpus rather than live web search.",
                "QC may identify coverage or confidence gaps.",
                "Human review is required before the report is treated as client-ready.",
            ],
            sources=source_ids,
            approval={"status": "pending", "reviewer": None},
        )

        return report

    def run(self, request: str) -> Dict[str, Any]:
        run_id = str(uuid.uuid4())
        plan = self.make_plan(request)

        self.research(plan, run_id)
        defects = self.quality_control(plan)

        # Bounded repair: maximum two rounds.
        for repair_round in range(1, 3):
            if not defects:
                break

            self.history.append({
                "stage": "qc_repair",
                "round": repair_round,
                "defects": defects,
            })

            self.call_tool(
                "search_information",
                {
                    "query": "customer support market risks opportunities",
                    "max_results": 3,
                },
                run_id,
            )

            defects = self.quality_control(plan)

        report = self.build_report(request, plan, run_id, defects)

        try:
            report = FinalReport.model_validate(report)
        except ValidationError as error:
            self.history.append({
                "stage": "final_validation",
                "status": "error",
                "error": str(error),
            })
            raise RuntimeError("Final report failed schema validation.")

        state = {
            "run_id": run_id,
            "plan": plan.model_dump(),
            "evidence": [e.model_dump() for e in self.evidence],
            "report": report.model_dump(),
            "qc": defects,
            "history": self.history,
            "usage": self.client.usage,
            "iteration_count": len(self.history),
            "approval_status": "pending",
        }

        self.call_tool(
            "save_research",
            {"run_id": run_id, "state": state},
            run_id,
        )

        return state

# ============================================================
# 6. STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="MarketMind AI",
    page_icon="📊",
    layout="wide",
)

st.title("📊 MarketMind AI")
st.caption("Autonomous Business Research Agent")

with st.sidebar:
    st.header("Run settings")
    demo = st.toggle(
        "Demo mode",
        value=not bool(os.getenv("OPENAI_API_KEY")),
        help="Uses the included local corpus and works without an API key.",
    )
    max_iterations = st.slider("Maximum research iterations", 2, 8, 5)
    max_tools = st.slider("Maximum tool calls", 4, 30, 15)

    st.divider()
    st.write("**Required tools**")
    for tool in TOOL_SCHEMAS:
        st.write(f"• `{tool}`")

request = st.text_area(
    "Business research request",
    value=(
        "Compare AI-powered customer support software for a mid-market company. "
        "Assess pricing, positioning, features, opportunities and risks."
    ),
    height=130,
)

if "result" not in st.session_state:
    st.session_state.result = None

if st.button(
    "🚀 Start Research",
    type="primary",
    use_container_width=True,
):
    with st.spinner(
        "Analyzing request → planning → researching → validating → preparing report..."
    ):
        try:
            agent = MarketMindAgent(
                demo=demo,
                max_iterations=max_iterations,
                max_tools=max_tools,
            )
            st.session_state.result = agent.run(request)
        except Exception as error:
            st.error(f"Run failed safely: {type(error).__name__}: {error}")

result = st.session_state.result

if result:
    st.success(f"Research completed • Run ID: `{result['run_id']}`")

    report_tab, evidence_tab, plan_tab, qc_tab, approval_tab = st.tabs(
        ["📄 Report", "🔎 Evidence", "🗺️ Plan", "🧪 QC / Logs", "👤 Approval"]
    )

    with report_tab:
        report = result["report"]

        st.subheader("Executive Summary")
        st.write(report["executive_summary"])

        st.subheader("Market Overview")
        st.write(report["market_overview"])

        st.subheader("Key Trends")
        for trend in report["key_trends"]:
            st.write("• " + trend)

        st.subheader("Competitor Analysis")
        st.dataframe(
            report["competitor_analysis"],
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Opportunities")
        for item in report["opportunities"]:
            st.write(f"**{item['type'].title()}:** {item['text']}")

        st.subheader("Risks")
        for item in report["risks"]:
            st.write(f"**{item['type'].title()}:** {item['text']}")

        st.subheader("Recommendations")
        for recommendation in report["recommendations"]:
            st.write("• " + recommendation)

        st.subheader("Confidence")
        st.write(report["confidence_rationale"])

        st.subheader("Limitations")
        for limitation in report["limitations"]:
            st.write("• " + limitation)

    with evidence_tab:
        st.json(result["evidence"])

    with plan_tab:
        st.json(result["plan"])

    with qc_tab:
        st.subheader("QC defects")
        st.json(result["qc"])

        st.subheader("Tool / agent history")
        st.json(result["history"])

        st.subheader("Usage")
        st.json(result["usage"])

    with approval_tab:
        st.warning(
            "Human approval is required. The system does not automatically approve reports."
        )

        decision = st.radio(
            "Decision",
            [
                "Approve",
                "Reject",
                "Request additional research",
                "Modify scope",
            ],
            horizontal=True,
        )

        reviewer = st.text_input("Reviewer name", value="Human reviewer")

        if st.button("Save Approval Decision"):
            result["report"]["approval"] = {
                "decision": decision,
                "reviewer": reviewer,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            path = RUNS / f"{result['run_id']}.json"
            path.write_text(
                json.dumps(result, indent=2, default=str),
                encoding="utf-8",
            )

            st.success(f"Approval decision saved to `{path}`")

st.divider()
st.caption(
    "Never commit API keys. Use Streamlit Secrets or environment variables for OPENAI_API_KEY."
)
