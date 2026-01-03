"""Weekly Strava training agent entry point."""

from __future__ import annotations

import os
import logging
import json
from pathlib import Path
from typing import Dict

from graph import run_once

__all__ = ["run_once"]


LOG_PATH = os.getenv("TRAINING_AGENT_LOG", "logs/training_agent.log")


def configure_logging(log_path: str | None = None) -> str:
    """Configure console + file logging for the agent."""
    path = Path(log_path or LOG_PATH)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        root_logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
        file_handler.setFormatter(fmt)
        root_logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(fmt)
        root_logger.addHandler(stream_handler)

    return str(path)


def _load_env() -> None:
    """Load .env if python-dotenv is available; otherwise continue."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ModuleNotFoundError:
        # If python-dotenv isn't installed, assume the environment was exported manually.
        pass


def _goal_from_env(default_goal: Dict) -> Dict:
    """Build goal from discrete env vars or fallback to JSON or default."""
    race = os.getenv("TRAINING_GOAL_RACE", "").strip()
    date = os.getenv("TRAINING_GOAL_DATE", "").strip()
    distance = os.getenv("TRAINING_GOAL_DISTANCE_KM", "").strip()
    elevation = os.getenv("TRAINING_GOAL_ELEVATION_GAIN_M", "").strip()

    if race or date or distance or elevation:
        goal: Dict = {
            "race": race or default_goal.get("race", ""),
            "date": date or default_goal.get("date", ""),
            "distance_km": float(distance) if distance else default_goal.get("distance_km", 0),
            "elevation_gain_m": float(elevation) if elevation else default_goal.get("elevation_gain_m", 0),
        }
        return goal

    raw_goal = os.getenv("TRAINING_GOAL_JSON", "").strip()
    if raw_goal:
        try:
            return json.loads(raw_goal)
        except json.JSONDecodeError:
            logging.getLogger(__name__).warning(
                "Failed to parse TRAINING_GOAL_JSON; falling back to defaults."
            )
    return default_goal


def main() -> None:
    # Load local environment so refresh credentials (client id/secret + refresh token)
    # are available when fetching a new Strava access token.
    _load_env()

    sample_goal: Dict = {
        "race": "Trail 25K",
        "date": "2025-09-01",
        "distance_km": 25,
        "elevation_gain_m": 800,
    }
    goal = _goal_from_env(sample_goal)
    log_file = configure_logging()
    logging.getLogger(__name__).info("Starting Strava training agent; logging to %s", log_file)

    run_once(
        goal=goal,
        sessions_per_week=int(os.getenv("SESSIONS_PER_WEEK", "3")),
        recipient_email=os.getenv("RECIPIENT_EMAIL", "lore@pairrot.eu"),
        # If STRAVA_ACCESS_TOKEN is absent, StravaClient will refresh using
        # STRAVA_CLIENT_ID + STRAVA_CLIENT_SECRET + STRAVA_REFRESH_TOKEN.
        strava_token=os.getenv("STRAVA_ACCESS_TOKEN"),
    )


if __name__ == "__main__":
    main()
