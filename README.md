# Strava Training Agent (LangGraph)

Weekly Strava training assistant that fetches activities, summarizes progress, generates a next-week plan with an LLM, and emails (or previews) the plan.

## Setup

1. Create/activate venv (optional):
   - `python3 -m venv env && source env/bin/activate`
2. Install deps:
   - `pip install langgraph langchain-openai langchain-core python-dotenv requests`
3. Configure environment in `.env` (or export):
   - Strava: `STRAVA_CLIENT_ID`, `STRAVA_CLIENT_SECRET`, `STRAVA_REFRESH_TOKEN` (auto-refreshes access token), optional `STRAVA_ACCESS_TOKEN`
   - OpenAI: `OPENAI_API_KEY`
   - Email (optional; previews if unset): `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_SENDER`, `RECIPIENT_EMAIL`
   - Other: `SESSIONS_PER_WEEK` (default 3), `TRAINING_AGENT_LOG` (default `logs/training_agent.log`), `TRAINING_AGENT_STATE_LOG` (default `logs/state_snapshots.log`)

## Run

```
python strava_training_agent.py
```

Behavior:

- Fetches Strava activities (auto-refresh token if creds present).
- Summarizes recent training and generates a plan (in km) via LLM.
- Adjusts plan if risks flagged, composes email, then:
  - Sends via SMTP when configured.
- Otherwise prints and logs the email preview.
- Renders `my_graph.png` (Mermaid) on each run.

## Schedule weekly via cron (example)

Run every Monday at 07:00 (adjust path/time as needed):

```
0 7 * * 1 cd /Users/lorevanoudenhove/Projects/tutorials/T202512_Langgraph_Strava && \
  source env/bin/activate && \
  python strava_training_agent.py >> logs/cron.log 2>&1
```

Notes:

- Ensure `.env` is present (or export env vars) so cron sees credentials.
- Keep the venv path and repo path matching your setup.
- Logs will accumulate in `logs/cron.log` (rotate as needed).

## Optional scheduling alternatives

- **GitHub Actions (low ops, no servers):**

  - Use a scheduled workflow with `on: schedule: cron: '0 7 * * 1'` (adjust).
  - Steps: checkout, set up Python, install deps, run `python strava_training_agent.py`.
  - Store Strava + SMTP + OpenAI secrets in GitHub Actions Secrets.
  - Good for lightweight jobs and easy logging.

- **AWS EventBridge + Lambda/ECS (production-grade):**
  - EventBridge Scheduler triggers weekly.
  - Lambda runs the weekly cycle; use Secrets Manager for Strava/SMTP/OpenAI secrets.
  - CloudWatch Logs for debugging. If deps are heavy, use an ECS Fargate scheduled task instead of Lambda.

## Logging

- Main log: `logs/training_agent.log` (set `TRAINING_AGENT_LOG` to override).
- State snapshots: `logs/state_snapshots.log` (set `TRAINING_AGENT_STATE_LOG`).
- LLM requests/responses and Strava payloads are logged with truncation safeguards.

## Graph overview

- Nodes: sync_strava_activities → summarize_recent_training → evaluate_progress_vs_goal → generate_next_week_plan → (conditional: keep → compose_weekly_email, adjust/deload → adjust_plan_add_warnings → compose_weekly_email) → send_email → END.
- Graph image saved as `my_graph.png` each run.
- Preview (generated): `![Workflow graph](my_graph.png)`
- Diagram source (mermaid): see `diagram.md`
