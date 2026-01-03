"""Graph wiring for the Strava training agent."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

from langgraph.graph import END, START, StateGraph

from nodes import (
    EmailClient,
    LLMService,
    StravaClient,
    adjust_plan_add_warnings,
    compose_weekly_email,
    configure_services,
    evaluate_progress_vs_goal,
    generate_next_week_plan,
    send_email,
    summarize_recent_training,
    sync_strava_activities,
)
from state import StravaTrainingAgentState, default_state

logger = logging.getLogger(__name__)


def build_graph(
    strava_client: StravaClient,
    llm: LLMService,
    email_client: EmailClient,
    recipient_email: str,
) -> StateGraph:
    """Build and compile the LangGraph workflow."""
    logger.info("Building training workflow graph.")
    configure_services(strava_client=strava_client, llm=llm, email_client=email_client, recipient_email=recipient_email)
    graph = StateGraph(StravaTrainingAgentState)
    graph.add_node("sync_strava_activities", sync_strava_activities)
    graph.add_node("summarize_recent_training", summarize_recent_training)
    graph.add_node("evaluate_progress_vs_goal", evaluate_progress_vs_goal)
    graph.add_node("generate_next_week_plan", generate_next_week_plan)
    graph.add_node("adjust_plan_add_warnings", adjust_plan_add_warnings)
    graph.add_node("compose_weekly_email", compose_weekly_email)
    graph.add_node("send_email", send_email)

    graph.add_edge(START, "sync_strava_activities")
    graph.add_edge("sync_strava_activities", "summarize_recent_training")
    graph.add_edge("summarize_recent_training", "evaluate_progress_vs_goal")
    graph.add_edge("evaluate_progress_vs_goal", "generate_next_week_plan")
    graph.add_conditional_edges(
        "generate_next_week_plan",
        lambda state: state.get("evaluation", {}).get("recommendation", "adjust"),
        {
            "keep": "compose_weekly_email",
            "adjust": "adjust_plan_add_warnings",
            "deload": "adjust_plan_add_warnings",
        },
    )
    graph.add_edge("adjust_plan_add_warnings", "compose_weekly_email")
    graph.add_edge("compose_weekly_email", "send_email")
    graph.add_edge("send_email", END)
    return graph


def run_once(
    goal: Dict,
    sessions_per_week: int,
    recipient_email: str,
    strava_token: Optional[str] = None,
) -> Dict:
    """Convenience runner for one weekly cycle."""
    # Configure logging if the caller did not do it already.
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        log_path = Path(os.getenv("TRAINING_AGENT_LOG", "logs/training_agent.log"))
        if log_path.parent:
            log_path.parent.mkdir(parents=True, exist_ok=True)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        root_logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(fmt)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(fmt)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(stream_handler)
        logger.info("Logging initialized at %s", log_path)

    strava_client = StravaClient(access_token=strava_token)
    llm = LLMService()
    email_client = EmailClient(
        smtp_host=os.getenv("SMTP_HOST", ""),
        smtp_port=int(os.getenv("SMTP_PORT", "587")),
        username=os.getenv("SMTP_USERNAME", ""),
        password=os.getenv("SMTP_PASSWORD", ""),
        sender=os.getenv("SMTP_SENDER", ""),
    )
    app = build_graph(strava_client, llm, email_client, recipient_email).compile()
    try:
        png_graph = app.get_graph().draw_mermaid_png()
        with open("my_graph.png", "wb") as f:
            f.write(png_graph)
        logger.info("Graph saved as 'my_graph.png' in %s", os.getcwd())
    except Exception as exc:
        logger.warning("Failed to render graph PNG: %s", exc)
    state = default_state(goal, sessions_per_week)
    logger.info(
        "Run %s initialized for goal '%s' with %d sessions/week.",
        state["run_id"],
        goal.get("race") or goal,
        sessions_per_week,
    )
    result = app.invoke(state)
    logger.info("Run %s completed.", state["run_id"])
    return result
