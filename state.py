"""State definitions for the Strava training agent."""

from __future__ import annotations

import uuid
from typing import Dict, List, Literal, TypedDict


class TrainingEvaluation(TypedDict):
    status: Literal["on_track", "behind", "overloaded"]
    confidence: float
    risk_flags: List[str]
    recommendation: Literal["keep", "adjust", "deload"]


class TrainingSession(TypedDict):
    day: str
    description: str
    duration_min: int
    intensity: Literal["easy", "moderate", "hard"]


class StravaTrainingAgentState(TypedDict):
    activities: List[Dict] | None
    training_summary: Dict | None
    goal: Dict
    sessions_per_week: int
    evaluation: TrainingEvaluation | None
    next_week_plan: List[TrainingSession] | None
    generated_plan: List[TrainingSession] | None
    weekly_email: str | None
    run_id: str
    last_sync_timestamp: str


def default_state(
    goal: Dict, sessions_per_week: int, last_sync_timestamp: str = ""
) -> StravaTrainingAgentState:
    """Create a baseline state for a new LangGraph run."""
    return StravaTrainingAgentState(
        activities=None,
        training_summary=None,
        goal=goal,
        sessions_per_week=sessions_per_week,
        evaluation=None,
        next_week_plan=None,
        generated_plan=None,
        weekly_email=None,
        run_id=str(uuid.uuid4()),
        last_sync_timestamp=last_sync_timestamp,
    )
