import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# ENVIRONMENT
# ============================================================

from dotenv import load_dotenv

load_dotenv(
    PROJECT_ROOT / ".env",
    override=True
)


# ============================================================
# THIRD-PARTY IMPORTS
# ============================================================

import pm4py

from src.analytics.process_analytics import (
    calculate_pr_cycle_time,
    calculate_transition_statistics,
)

from src.rag.retriever import ProcessRetriever

from src.agents.groq_agent import GroqProcessAgent


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Agentic Process Intelligence",
    page_icon="🔎",
    layout="wide",
)


# ============================================================
# TITLE
# ============================================================

st.title("🔎 Agentic Process Intelligence")

st.caption(
    "AI-powered Purchase-to-Pay process analytics, "
    "predictive risk, RAG and agentic reasoning"
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_process_data():

    db_path = (
        PROJECT_ROOT
        / "data"
        / "lrms_02_p2p.sqlite"
    )

    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found: {db_path}"
        )

    # Read OCEL 2.0 database
    ocel = pm4py.read_ocel2(
        str(db_path)
    )

    # Extract raw tables
    events = ocel.events.copy()
    objects = ocel.objects.copy()
    relations = ocel.relations.copy()

    return (
        events,
        relations,
        objects,
    )


try:

    (
        events,
        relations,
        objects,
    ) = load_process_data()

except Exception as exc:

    st.error(
        f"Unable to load OCEL database: {exc}"
    )

    st.stop()


# ============================================================
# BUILD EVENT-OBJECT TABLE
# ============================================================

@st.cache_data
def build_pr_relations(
    events,
    relations,
):

    # Keep PR objects
    pr_relations = relations[
        relations["ocel:type"]
        == "Purchase Requisition"
    ].copy()

    # Add event context
    event_context = events[
        [
            "ocel:eid",
            "user",
            "cost_center",
            "ocel:activity",
            "ocel:timestamp",
        ]
    ].copy()

    pr_relations = pr_relations.merge(
        event_context,
        on="ocel:eid",
        how="left",
        suffixes=(
            "",
            "_event",
        ),
    )

    # Clean analytical fields
    if "ocel:activity_event" in pr_relations.columns:

        pr_relations["activity"] = (
            pr_relations[
                "ocel:activity_event"
            ]
        )

    else:

        pr_relations["activity"] = (
            pr_relations[
                "ocel:activity"
            ]
        )

    if "ocel:timestamp_event" in pr_relations.columns:

        pr_relations["timestamp"] = pd.to_datetime(
            pr_relations[
                "ocel:timestamp_event"
            ]
        )

    else:

        pr_relations["timestamp"] = pd.to_datetime(
            pr_relations[
                "ocel:timestamp"
            ]
        )

    # Also make sure analytics functions
    # have the expected OCEL fields.
    pr_relations["ocel:activity"] = (
        pr_relations["activity"]
    )

    pr_relations["ocel:timestamp"] = (
        pr_relations["timestamp"]
    )

    return pr_relations


try:

    pr_relations = build_pr_relations(
        events,
        relations,
    )

except Exception as exc:

    st.error(
        f"Failed to build PR event table: {exc}"
    )

    st.stop()


# ============================================================
# PROCESS ANALYTICS
# ============================================================

try:

    pr_cycle = calculate_pr_cycle_time(
        pr_relations
    )

    transition_stats = (
        calculate_transition_statistics(
            pr_relations
        )
    )

except Exception as exc:

    st.error(
        f"Process analytics failed: {exc}"
    )

    st.stop()


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_prs = len(pr_cycle)

average_cycle = (
    pr_cycle["cycle_time_hours"].mean()
)

median_cycle = (
    pr_cycle["cycle_time_hours"].median()
)

p90_cycle = (
    pr_cycle["cycle_time_hours"].quantile(
        0.90
    )
)


# ============================================================
# REJECTION METRICS
# ============================================================

rejected_pr_ids = set(
    pr_relations.loc[
        pr_relations["activity"]
        == "Reject PR",
        "ocel:oid",
    ]
)

rejected_prs = len(
    rejected_pr_ids
)

approved_prs = (
    total_prs
    - rejected_prs
)

approval_rate = (
    approved_prs
    / total_prs
    * 100
)

rejection_rate = (
    rejected_prs
    / total_prs
    * 100
)


# ============================================================
# THREE-WAY MATCH METRICS
# ============================================================

match_events = events[
    events["ocel:activity"]
    .astype(str)
    .str.contains(
        "Three.?way Match",
        case=False,
        na=False,
        regex=True,
    )
].copy()


successful_matches = int(
    match_events[
        "ocel:activity"
    ]
    .astype(str)
    .str.contains(
        "Successful",
        case=False,
        na=False,
    )
    .sum()
)


mismatch_count = int(
    match_events[
        "ocel:activity"
    ]
    .astype(str)
    .str.contains(
        "Mismatch",
        case=False,
        na=False,
    )
    .sum()
)


total_matches = (
    successful_matches
    + mismatch_count
)


if total_matches > 0:

    match_success_rate = (
        successful_matches
        / total_matches
        * 100
    )

    mismatch_rate = (
        mismatch_count
        / total_matches
        * 100
    )

else:

    match_success_rate = 0.0
    mismatch_rate = 0.0


# ============================================================
# MULTIPLE PO CREATION
# ============================================================

po_creation_counts = (
    pr_relations[
        pr_relations["activity"].isin(
            [
                "Create PO",
                "Create PO (Batch)",
            ]
        )
    ]
    .groupby("ocel:oid")
    .size()
)

multiple_po_prs = int(
    (
        po_creation_counts > 1
    ).sum()
)


# ============================================================
# LOAD ML RISK DATA
# ============================================================

risk_path = (
    PROJECT_ROOT
    / "data"
    / "pr_intelligence.csv"
)

if risk_path.exists():

    risk_df = pd.read_csv(
        risk_path
    )

else:

    risk_df = pd.DataFrame()


# ============================================================
# RAG
# ============================================================

@st.cache_resource
def load_retriever():

    return ProcessRetriever(
        PROJECT_ROOT
    )


try:

    retriever = load_retriever()

except Exception:

    retriever = None


# ============================================================
# AGENT TOOLS
# ============================================================

def get_kpis():

    return {
        "total_prs": int(
            total_prs
        ),
        "approved_prs": int(
            approved_prs
        ),
        "rejected_prs": int(
            rejected_prs
        ),
        "approval_rate": round(
            approval_rate,
            2,
        ),
        "rejection_rate": round(
            rejection_rate,
            2,
        ),
        "average_cycle_time_hours": round(
            average_cycle,
            3,
        ),
        "median_cycle_time_hours": round(
            median_cycle,
            3,
        ),
        "p90_cycle_time_hours": round(
            p90_cycle,
            3,
        ),
        "successful_matches": int(
            successful_matches
        ),
        "mismatches": int(
            mismatch_count
        ),
        "match_success_rate": round(
            match_success_rate,
            2,
        ),
        "mismatch_rate": round(
            mismatch_rate,
            2,
        ),
        "prs_with_multiple_po_creation_events": int(
            multiple_po_prs
        ),
    }


def get_bottlenecks(
    top_n: int = 5
):

    top_n = int(top_n)

    bottlenecks = (
        transition_stats
        .sort_values(
            "average_hours",
            ascending=False,
        )
        .head(top_n)
    )

    results = []

    for _, row in bottlenecks.iterrows():

        results.append(
            {
                "transition": (
                    f"{row['previous_activity']} "
                    f"-> "
                    f"{row['ocel:activity']}"
                ),
                "occurrences": int(
                    row["occurrences"]
                ),
                "average_hours": round(
                    float(
                        row["average_hours"]
                    ),
                    3,
                ),
                "median_hours": round(
                    float(
                        row["median_hours"]
                    ),
                    3,
                ),
                "p90_hours": round(
                    float(
                        row["p90_hours"]
                    ),
                    3,
                ),
                "max_hours": round(
                    float(
                        row["max_hours"]
                    ),
                    3,
                ),
            }
        )

    return results


def get_exceptions():

    return {
        "rejected_prs": rejected_prs,
        "rejection_rate": round(
            rejection_rate,
            2,
        ),
        "prs_with_multiple_po_creation_events": (
            multiple_po_prs
        ),
        "three_way_match_mismatches": (
            mismatch_count
        ),
        "three_way_match_mismatch_rate": round(
            mismatch_rate,
            2,
        ),
    }


def get_pr_risk(
    pr_id: str
):

    if risk_df.empty:

        return {
            "error": (
                "Risk dataset is not available."
            )
        }

    matches = risk_df[
        risk_df["ocel:oid"]
        .astype(str)
        .str.upper()
        == str(pr_id).upper()
    ]

    if matches.empty:

        return {
            "error": (
                f"No risk information found "
                f"for {pr_id}."
            )
        }

    row = matches.iloc[0]

    return {
        "pr_id": str(
            row["ocel:oid"]
        ),
        "user": str(
            row["user"]
        ),
        "cost_center": str(
            row["cost_center"]
        ),
        "slow_pr_probability": round(
            float(
                row["slow_pr_probability"]
            ),
            4,
        ),
        "risk_band": str(
            row["risk_band"]
        ),
        "risk_explanation": str(
            row["risk_explanation"]
        ),
    }


def search_process_knowledge(
    question: str
):

    if retriever is None:

        return {
            "error": (
                "RAG retriever is unavailable."
            )
        }

    results = retriever.retrieve(
        question,
        top_k=5,
    )

    evidence = []

    for item in results:

        evidence.append(
            {
                "source": item[
                    "metadata"
                ].get(
                    "source",
                    "unknown",
                ),
                "document": item[
                    "document"
                ],
                "distance": item.get(
                    "distance"
                ),
            }
        )

    return evidence


# ============================================================
# GROQ AGENT
# ============================================================

groq_api_key = os.getenv(
    "GROQ_API_KEY"
)

agent = None

if groq_api_key:

    try:

        agent = GroqProcessAgent(
            get_kpis=get_kpis,
            get_bottlenecks=get_bottlenecks,
            get_exceptions=get_exceptions,
            get_pr_risk=get_pr_risk,
            search_process_knowledge=(
                search_process_knowledge
            ),
        )

    except Exception as exc:

        st.warning(
            f"Agent initialization failed: {exc}"
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "🔎 Process Intelligence"
    )

    st.markdown(
        """
### Pipeline

OCEL 2.0

↓

Process Analytics

↓

Predictive ML

↓

RAG

↓

Groq Agent
"""
    )

    st.divider()

    st.subheader(
        "Dataset"
    )

    st.write(
        f"Events: {len(events):,}"
    )

    st.write(
        f"Relations: {len(relations):,}"
    )

    st.write(
        f"Objects: {len(objects):,}"
    )


# ============================================================
# KPI CARDS
# ============================================================

st.subheader(
    "P2P Process Overview"
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Total PRs",
        f"{total_prs:,}",
    )

with col2:

    st.metric(
        "Avg Cycle Time",
        f"{average_cycle:.2f} h",
    )

with col3:

    st.metric(
        "Rejection Rate",
        f"{rejection_rate:.1f}%",
    )

with col4:

    st.metric(
        "3-Way Mismatch",
        f"{mismatch_rate:.1f}%",
    )


# ============================================================
# TABS
# ============================================================

overview_tab, bottleneck_tab, risk_tab, agent_tab = st.tabs(
    [
        "📊 Overview",
        "⏱️ Bottlenecks",
        "🤖 Risk Analysis",
        "💬 AI Agent",
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with overview_tab:

    st.subheader(
        "Process KPIs"
    )

    overview_data = pd.DataFrame(
        {
            "Metric": [
                "Total PRs",
                "Approved PRs",
                "Rejected PRs",
                "Approval Rate",
                "Rejection Rate",
                "Average Cycle Time",
                "Median Cycle Time",
                "P90 Cycle Time",
                "Successful Matches",
                "Match Mismatches",
                "Mismatch Rate",
                "PRs with Multiple PO-Creation Events",
            ],
            "Value": [
                total_prs,
                approved_prs,
                rejected_prs,
                f"{approval_rate:.2f}%",
                f"{rejection_rate:.2f}%",
                f"{average_cycle:.3f} h",
                f"{median_cycle:.3f} h",
                f"{p90_cycle:.3f} h",
                successful_matches,
                mismatch_count,
                f"{mismatch_rate:.2f}%",
                multiple_po_prs,
            ],
        }
    )

    st.dataframe(
        overview_data,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Multiple PO-creation events are reported as a "
        "process pattern. They are not automatically treated "
        "as rework without additional business evidence."
    )


# ============================================================
# BOTTLENECKS
# ============================================================

with bottleneck_tab:

    st.subheader(
        "Top Process Bottlenecks"
    )

    bottleneck_df = (
        transition_stats
        .sort_values(
            "average_hours",
            ascending=False,
        )
        .head(10)
        .copy()
    )

    bottleneck_df[
        "transition"
    ] = (
        bottleneck_df[
            "previous_activity"
        ].astype(str)
        + " → "
        + bottleneck_df[
            "ocel:activity"
        ].astype(str)
    )

    bottleneck_display = bottleneck_df[
        [
            "transition",
            "occurrences",
            "average_hours",
            "median_hours",
            "p90_hours",
            "max_hours",
        ]
    ].copy()

    bottleneck_display.columns = [
        "Transition",
        "Occurrences",
        "Average (h)",
        "Median (h)",
        "P90 (h)",
        "Maximum (h)",
    ]

    st.dataframe(
        bottleneck_display,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# RISK ANALYSIS
# ============================================================

with risk_tab:

    st.subheader(
        "Predictive PR Risk"
    )

    if risk_df.empty:

        st.warning(
            "data/pr_intelligence.csv not found."
        )

    else:

        risk_display = risk_df[
            [
                "ocel:oid",
                "user",
                "cost_center",
                "slow_pr_probability",
                "risk_band",
                "risk_explanation",
            ]
        ].copy()

        risk_display.columns = [
            "PR",
            "User",
            "Cost Center",
            "Slow-PR Score",
            "Risk Band",
            "Explanation",
        ]

        st.dataframe(
            risk_display,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# AI AGENT
# ============================================================

with agent_tab:

    st.subheader(
        "💬 Ask the Process Intelligence Agent"
    )

    st.markdown(
        """
The agent can analyze:

**KPIs** • **Bottlenecks** • **Exceptions** •
**PR Risk** • **Process Knowledge**
"""
    )

    if "messages" not in st.session_state:

        st.session_state.messages = []


    # Previous messages
    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    question = st.chat_input(
        "Ask a process question..."
    )


    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):

            st.markdown(
                question
            )


        with st.chat_message("assistant"):

            if agent is None:

                answer = (
                    "Groq is not configured. "
                    "Check GROQ_API_KEY in your .env file."
                )

            else:

                with st.spinner(
                    "Analyzing process data..."
                ):

                    try:

                        answer = agent.run(
                            question
                        )

                    except Exception as exc:

                        answer = (
                            "The agent encountered an error:\n\n"
                            f"{exc}"
                        )

            st.markdown(
                answer
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "OCEL 2.0 • Process Analytics • Predictive ML • "
    "RAG • Groq Tool-Calling Agent"
)