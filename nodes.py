"""Node implementations and service clients for the Strava training agent."""

from __future__ import annotations

import logging
import json
import os
import smtplib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import Dict, List, Optional

try:
    # Load .env so Strava credentials are available even when caller forgets.
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # Optional dependency; we still proceed if not installed.
    pass

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
except ImportError:  # Optional dependency; runtime will raise if LLM is required.
    ChatOpenAI = None
    HumanMessage = None
    SystemMessage = None

from state import StravaTrainingAgentState, TrainingEvaluation, TrainingSession

logger = logging.getLogger(__name__)

# Limit payload size in logs to avoid runaway files.
_MAX_LOG_CHARS = 8000
_STATE_LOG_PATH = os.getenv("TRAINING_AGENT_STATE_LOG", "logs/state_snapshots.log")
_state_logger: Optional[logging.Logger] = None
_services: Dict[str, object] = {
    "strava_client": None,
    "llm": None,
    "email_client": None,
    "recipient_email": None,
}


def _get_state_logger() -> logging.Logger:
    global _state_logger
    if _state_logger is not None:
        return _state_logger
    state_logger = logging.getLogger("training_agent.state")
    state_logger.propagate = False
    fmt = logging.Formatter("%(asctime)s [STATE] %(message)s")
    try:
        os.makedirs(os.path.dirname(_STATE_LOG_PATH), exist_ok=True)
    except Exception:
        # If directory creation fails, continue with default location.
        pass
    try:
        handler = logging.FileHandler(_STATE_LOG_PATH, mode="a", encoding="utf-8")
        handler.setFormatter(fmt)
        state_logger.addHandler(handler)
        state_logger.setLevel(logging.INFO)
    except Exception:
        # If file handler fails, fall back to stderr.
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(fmt)
        state_logger.addHandler(stream_handler)
        state_logger.setLevel(logging.INFO)
    _state_logger = state_logger
    return state_logger


def _log_payload(label: str, payload: object, max_chars: int = _MAX_LOG_CHARS) -> None:
    """Serialize payload to JSON-ish text with truncation and log at info level."""
    try:
        text = json.dumps(payload, default=str)
    except Exception:
        text = str(payload)
    truncated = len(text) > max_chars
    display = text[:max_chars]
    if truncated:
        display += " ...[truncated]"
    logger.info("%s: %s", label, display)


def _log_state(label: str, state: Dict) -> None:
    """Write full state snapshots to the dedicated state log file."""
    try:
        text = json.dumps(state, default=str)
    except Exception:
        text = str(state)
    state_logger = _get_state_logger()
    state_logger.info("%s: %s", label, text)


def configure_services(
    strava_client: "StravaClient",
    llm: "LLMService",
    email_client: "EmailClient",
    recipient_email: str,
) -> None:
    """Inject shared services for node functions to consume directly."""
    _services["strava_client"] = strava_client
    _services["llm"] = llm
    _services["email_client"] = email_client
    _services["recipient_email"] = recipient_email


def _get_service(name: str):
    service = _services.get(name)
    if service is None:
        raise RuntimeError(f"Service '{name}' has not been configured.")
    return service


@dataclass
class StravaClient:
    """Thin Strava REST client."""

    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None

    def __post_init__(self) -> None:
        self.access_token = self.access_token or os.getenv("STRAVA_ACCESS_TOKEN")
        self.refresh_token = self.refresh_token or os.getenv("STRAVA_REFRESH_TOKEN")
        self.client_id = self.client_id or os.getenv("STRAVA_CLIENT_ID")
        self.client_secret = self.client_secret or os.getenv("STRAVA_CLIENT_SECRET")

        can_refresh = self.refresh_token and self.client_id and self.client_secret
        if can_refresh:
            logger.info("Refreshing Strava access token at startup.")
            # Always refresh at startup when credentials are present to avoid using expired tokens.
            self.access_token = self._refresh_access_token()

        if not self.access_token:
            raise RuntimeError(
                "STRAVA_ACCESS_TOKEN is required; provide it directly or set refresh credentials."
            )
        logger.info("Strava client ready; access token present.")

    def _refresh_access_token(self) -> str:
        import requests

        logger.info("Requesting new Strava access token.")
        payload = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
        }
        response = requests.post("https://www.strava.com/oauth/token", data=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        token = data.get("access_token")
        if not token:
            raise RuntimeError("Failed to refresh Strava access token.")
        logger.info("Strava access token refreshed successfully.")
        # Persist refreshed tokens in the process env for downstream calls.
        os.environ["STRAVA_ACCESS_TOKEN"] = token
        new_refresh = data.get("refresh_token")
        if new_refresh:
            os.environ["STRAVA_REFRESH_TOKEN"] = new_refresh
        return token

    def fetch_activities(self, since: datetime) -> List[Dict]:
        import requests

        logger.info("Fetching activities since %s", since.isoformat())
        url = "https://www.strava.com/api/v3/athlete/activities"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        params = {"after": int(since.timestamp()), "per_page": 200}
        activities: List[Dict] = []
        page = 1
        while True:
            page_params = {**params, "page": page}
            response = requests.get(url, headers=headers, params=page_params, timeout=20)
            if response.status_code == 401 and self.refresh_token:
                # Token likely expired; refresh once and retry current page.
                self.access_token = self._refresh_access_token()
                headers["Authorization"] = f"Bearer {self.access_token}"
                response = requests.get(url, headers=headers, params=page_params, timeout=20)
            if response.status_code == 401:
                detail = ""
                try:
                    detail = response.json()
                except Exception:
                    detail = response.text
                raise RuntimeError(
                    "Strava returned 401 Unauthorized when fetching activities. "
                    "Confirm STRAVA_CLIENT_ID/STRAVA_CLIENT_SECRET/STRAVA_REFRESH_TOKEN are valid "
                    "and that the token has the 'activity:read_all' scope. "
                    f"Response detail: {detail}"
                )
            response.raise_for_status()
            batch = response.json()
            activities.extend(batch)
            if len(batch) < params["per_page"]:
                break
            page += 1
        logger.info("Fetched %d activities from Strava.", len(activities))
        _log_payload("strava_activities", activities)
        return activities


@dataclass
class LLMService:
    """Wrapper for chat completions."""

    model: str = "gpt-4o-mini"
    temperature: float = 0.1

    def __post_init__(self) -> None:
        if ChatOpenAI is None:
            raise RuntimeError("langchain_openai is required for LLM calls.")
        self.client = ChatOpenAI(model=self.model, temperature=self.temperature)

    def structured_completion(self, system: str, user: str) -> str:
        _log_payload("llm_request", {"system": system, "user": user})
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=user),
        ]
        result = self.client.invoke(messages)
        _log_payload("llm_response", result.content)
        return result.content


@dataclass
class EmailClient:
    """SMTP email sender."""

    smtp_host: str
    smtp_port: int
    username: str
    password: str
    sender: str

    def send(self, to_email: str, subject: str, body: str) -> None:
        preview_msg = (
            f"Subject: {subject}\nTo: {to_email or '(missing recipient)'}\n\n{body}"
        )
        # Show preview and skip sending if SMTP is not configured.
        if not self.smtp_host:
            logger.info(
                "SMTP disabled: showing email preview instead. Set SMTP_HOST/SMTP_PORT/SMTP_USERNAME/SMTP_PASSWORD/SMTP_SENDER to enable email sending."
            )
            logger.info("Email preview:\n%s", preview_msg)
            print(preview_msg)
            return
        if not to_email:
            logger.warning("No recipient email provided; showing email preview instead of sending.")
            logger.info("Email preview:\n%s", preview_msg)
            print(preview_msg)
            return
        msg = EmailMessage()
        msg["From"] = self.sender
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.set_content(body)

        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
        logger.info("Email delivered to %s.", to_email)


def parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sync_strava_activities(
    state: StravaTrainingAgentState
) -> Dict:
    """Fetch Strava activities from the last 90 days."""
    _log_state("before_sync_strava_activities", state)
    client: StravaClient = _get_service("strava_client")  # type: ignore[assignment]
    since = datetime.now(timezone.utc) - timedelta(days=90)

    logger.info("Syncing Strava activities (last 90 days) since %s", since.isoformat())
    activities = client.fetch_activities(since=since)
    logger.info("Synced %d activities.", len(activities))
    result = {"activities": activities}
    _log_state("after_sync_strava_activities", {**state, **result})
    return result


def summarize_recent_training(state: StravaTrainingAgentState) -> Dict:
    """Aggregate raw activities into recent training signals."""
    _log_state("before_summarize_recent_training", state)
    activities = state.get("activities") or []
    now = datetime.now(timezone.utc)
    seven_days_ago = now - timedelta(days=7)
    recent = []
    for act in activities:
        start_date = act.get("start_date_local") or act.get("start_date")
        if not start_date:
            continue
        try:
            start_dt = parse_ts(start_date)
        except ValueError:
            continue
        if start_dt >= seven_days_ago:
            recent.append((act, start_dt))

    weekly_minutes = int(
        sum((a.get("moving_time") or a.get("elapsed_time") or 0) / 60 for a, _ in recent)
    )
    weekly_km = round(sum((a.get("distance") or 0) / 1000 for a, _ in recent), 2)

    def is_hard(activity: Dict) -> bool:
        if activity.get("suffer_score", 0) >= 70:
            return True
        if activity.get("average_heartrate", 0) >= 160:
            return True
        return activity.get("workout_type") in {2, 3}

    hard_sessions = [a for a, _ in recent if is_hard(a)]
    longest = None
    if recent:
        longest = max(
            recent,
            key=lambda pair: pair[0].get("moving_time") or pair[0].get("elapsed_time") or 0,
        )[0]

    consistency = 0.0
    if state.get("sessions_per_week"):
        consistency = round(len(recent) / float(state["sessions_per_week"]), 2)

    summary = {
        "weekly_volume_minutes": weekly_minutes,
        "weekly_volume_km": weekly_km,
        "hard_sessions": len(hard_sessions),
        "longest_session": {
            "name": longest.get("name") if longest else None,
            "duration_min": int(
                (longest.get("moving_time") or longest.get("elapsed_time") or 0) / 60
            )
            if longest
            else None,
            "distance_km": round((longest.get("distance") or 0) / 1000, 2) if longest else None,
        },
        "consistency_ratio": consistency,
        "session_count": len(recent),
        "recency_window_days": 7,
    }
    logger.info("Training summary computed: %s", summary)
    result = {"training_summary": summary}
    _log_state("after_summarize_recent_training", {**state, **result})
    return result


def _safe_json_loads(content: str) -> Dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Handle common LLM formatting with fenced JSON blocks.
        fenced = content.strip()
        if fenced.startswith("```"):
            fenced = fenced.strip("`")
            if fenced.lstrip().startswith("json"):
                # Drop optional language hint
                fenced = fenced[fenced.find("\n") + 1 :]
        try:
            return json.loads(fenced)
        except Exception:
            return {}


def evaluate_progress_vs_goal(
    state: StravaTrainingAgentState
) -> Dict:
    """Compare training summary with the goal using an LLM."""
    _log_state("before_evaluate_progress_vs_goal", state)
    llm: LLMService = _get_service("llm")  # type: ignore[assignment]
    summary = state.get("training_summary") or {}
    goal = state.get("goal") or {}
    sessions_per_week = state.get("sessions_per_week")

    system_prompt = (
        "You are a physical training reviewer. Evaluate progress toward the goal with short JSON.\n"
        "Use overload heuristics: sudden volume spikes >25% week-over-week, stacking hard sessions,\n"
        "or excessive long runs. Avoid medical language. Do not include explanations."
    )
    user_prompt = (
        f"Goal:\n{json.dumps(goal, indent=2)}\n\n"
        f"Target sessions per week: {sessions_per_week}\n"
        f"Training summary:\n{json.dumps(summary, indent=2)}\n\n"
        "Return JSON with keys: status (on_track|behind|overloaded), confidence (0-1), "
        "risk_flags (array), recommendation (keep|adjust|deload)."
    )
    response = llm.structured_completion(system_prompt, user_prompt)
    evaluation = _safe_json_loads(response)

    fallback: TrainingEvaluation = {
        "status": "behind",
        "confidence": 0.4,
        "risk_flags": ["llm_parse_failed"],
        "recommendation": "adjust",
    }
    chosen = evaluation or fallback
    logger.info("Evaluation result: %s", chosen)
    result = {"evaluation": chosen}
    _log_state("after_evaluate_progress_vs_goal", {**state, **result})
    return result


def generate_next_week_plan(
    state: StravaTrainingAgentState
) -> Dict:
    """Create the next week of training using the chosen strategy."""
    _log_state("before_generate_next_week_plan", state)
    llm: LLMService = _get_service("llm")  # type: ignore[assignment]
    summary = state.get("training_summary") or {}
    goal = state.get("goal") or {}
    sessions = state.get("sessions_per_week") or 3

    system_prompt = (
        "You are a running coach. Propose one-week plan as JSON list of sessions. "
        "Respect sessions_per_week exactly. Cap progression at +10% volume unless deload "
        "requires 20-30% reduction. Prefer goal-specific terrain. Keep descriptions concise. "
        "Use kilometers (km) for any distance reference."
    )
    user_prompt = (
        f"Goal: {json.dumps(goal)}\n"
        f"Sessions per week: {sessions}\n"
        f"Recent summary: {json.dumps(summary)}\n"
        "Return a JSON array of objects with fields: day (Mon-Sun), description, duration_min, intensity (easy|moderate|hard)."
    )
    response = llm.structured_completion(system_prompt, user_prompt)
    plan = _safe_json_loads(response)
    if not isinstance(plan, list):
        plan = []
    normalized: List[TrainingSession] = []
    for item in plan:
        if not isinstance(item, dict):
            continue
        normalized.append(
            TrainingSession(
                day=str(item.get("day", "")),
                description=str(item.get("description", "")),
                duration_min=int(item.get("duration_min", 0)),
                intensity=item.get("intensity", "easy"),
            )
        )

    target_sessions = sessions if sessions > 0 else len(normalized)
    trimmed_plan = normalized[:target_sessions]
    logger.info(
        "Generated next week plan with %d sessions (requested=%d).",
        len(trimmed_plan),
        sessions,
    )
    result = {"next_week_plan": trimmed_plan, "generated_plan": trimmed_plan}
    _log_state("after_generate_next_week_plan", {**state, **result})
    return result


def adjust_plan_add_warnings(
    state: StravaTrainingAgentState
) -> Dict:
    """Soften the plan and craft warning text when risk flags exist."""
    _log_state("before_adjust_plan_add_warnings", state)
    llm: LLMService = _get_service("llm")  # type: ignore[assignment]
    plan = state.get("next_week_plan") or []
    generated_plan = state.get("generated_plan") or plan
    evaluation = state.get("evaluation") or {}
    risk_flags = evaluation.get("risk_flags") or []
    if not risk_flags:
        logger.info("No risk flags detected; plan kept as-is.")
        return {"next_week_plan": plan, "generated_plan": generated_plan}

    system_prompt = (
        "You are a safety-focused coach. If risks exist, gently soften the plan while keeping "
        "the number of sessions the same. Keep text brief and avoid medical phrasing. "
        "Use kilometers (km) for any distance reference."
    )
    user_prompt = (
        f"Risks: {risk_flags}\n"
        f"Current plan: {json.dumps(plan)}\n"
        "Return JSON with key next_week_plan (array of sessions) and short_warning (string)."
    )
    response = llm.structured_completion(system_prompt, user_prompt)
    parsed = _safe_json_loads(response)
    updated_plan = parsed.get("next_week_plan") if isinstance(parsed, dict) else None
    warning_note = parsed.get("short_warning") if isinstance(parsed, dict) else None

    final_plan: List[TrainingSession] = []
    if isinstance(updated_plan, list) and updated_plan:
        for item in updated_plan:
            if not isinstance(item, dict):
                continue
            final_plan.append(
                TrainingSession(
                    day=str(item.get("day", "")),
                    description=str(item.get("description", "")),
                    duration_min=int(item.get("duration_min", 0)),
                    intensity=item.get("intensity", "easy"),
                )
            )
    else:
        final_plan = plan

    if warning_note:
        new_flags = list(risk_flags)
        new_flags.append(f"warning_note: {warning_note}")
        patched_eval = {
            **evaluation,
            "risk_flags": new_flags,
        }
        logger.info("Plan softened due to risks %s with note: %s", risk_flags, warning_note)
        result = {"next_week_plan": final_plan, "evaluation": patched_eval, "generated_plan": generated_plan}
        _log_state("after_adjust_plan_add_warnings", {**state, **result})
        return result
    logger.info("Plan adjusted due to risks %s", risk_flags)
    result = {"next_week_plan": final_plan, "generated_plan": generated_plan}
    _log_state("after_adjust_plan_add_warnings", {**state, **result})
    return result


def compose_weekly_email(
    state: StravaTrainingAgentState
) -> Dict:
    """Compose the weekly email with plan, rationale, and warnings."""
    _log_state("before_compose_weekly_email", state)
    llm: LLMService = _get_service("llm")  # type: ignore[assignment]
    plan = state.get("next_week_plan") or []
    generated_plan = state.get("generated_plan") or plan
    evaluation = state.get("evaluation") or {}
    summary = state.get("training_summary") or {}

    system_prompt = (
        "You are a concise coach. Write a weekly email with intro, rationale, plan, and warnings. "
        "Keep it friendly, direct, and avoid medical advice. Use kilometers (km) for any distance reference."
    )
    user_prompt = (
        f"Evaluation: {json.dumps(evaluation)}\n"
        f"Summary: {json.dumps(summary)}\n"
        f"Plan: {json.dumps(plan)}\n"
        "Warnings: derive from risk_flags only if present.\n"
        "Return the full email as plain text. Keep it under 200 words."
    )
    email_body = llm.structured_completion(system_prompt, user_prompt)
    logger.info("Weekly email composed; length=%d characters.", len(email_body))

    def format_plan(sessions: List[TrainingSession]) -> str:
        lines = []
        for sess in sessions:
            lines.append(
                f"- {sess.get('day')}: {sess.get('description')} "
                f"({sess.get('duration_min')} min, {sess.get('intensity')})"
            )
        return "\n".join(lines)

    plan_section = format_plan(plan)
    generated_section = format_plan(generated_plan)
    full_email = email_body
    result = {"weekly_email": full_email}
    _log_state("after_compose_weekly_email", {**state, **result})
    return result


def send_email(
    state: StravaTrainingAgentState,
    subject_prefix: str = "Weekly Training Plan",
) -> Dict:
    """Send the composed weekly email."""
    _log_state("before_send_email", state)
    email_client: EmailClient = _get_service("email_client")  # type: ignore[assignment]
    recipient: str = _get_service("recipient_email")  # type: ignore[assignment]
    body = state.get("weekly_email")
    if not body:
        logger.warning("No email body generated; skipping send.")
        return {}
    subject = f"{subject_prefix} - {state.get('run_id')}"
    logger.info("Sending weekly email to %s with subject '%s'.", recipient, subject)
    email_client.send(recipient, subject, body)
    _log_state("after_send_email", state)
    return {}
