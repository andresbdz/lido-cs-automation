# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Lido CS Automation is a webhook-based system that processes tldv.io meeting recordings, extracts insights using Claude AI, and updates Google Sheets with structured customer success data.

**Stack:** Python 3.10+ / FastAPI / Anthropic Claude / Google Sheets API / tldv.io API
**Deployed:** https://lido-cs-automation-production.up.railway.app

## Commands

```bash
# Run locally
uvicorn app:app --reload --port 8000

# Run tests
pytest tests/
pytest tests/test_duplicate_prevention.py -v
```

## Architecture

### Data Flow
1. **Trigger** → tldv webhook (`/webhook/tldv`) OR polling fallback (every 15 min)
2. **Classify** → CS / Sales / Skip based on meeting title keywords
3. **Analyze** → Claude extracts: next steps, Q&A pairs, follow-up email, marketing worthiness, volume
4. **Store** → Google Sheets (CS tab or Sales tab) + knowledge_base.md

### Key Files
- `app.py` - FastAPI app, webhook handlers, processing pipeline, duplicate prevention
- `services/tldv_client.py` - tldv API integration, recording classification
- `services/transcript_analyzer.py` - Claude AI extraction (next steps, Q&A, emails)
- `services/sheets_client.py` - Google Sheets API with dual auth (file or env var)
- `services/knowledge_base_writer.py` - Local markdown Q&A storage

### Duplicate Prevention
Atomic claiming via `try_claim_recording()` prevents race conditions. Recordings are tracked in `processed_recordings.json` with statuses: processing → completed. Stale "processing" entries (>10 min) are reclaimed.

### Recording Classification
- **CS tab:** title contains "customer success", "check-in", "weekly", "monthly"
- **Skip:** title contains "internal", "standup", "stand-up"
- **Sales tab:** everything else (default)

## Environment Variables

```
TLDV_API_KEY                    # Required
ANTHROPIC_API_KEY               # Required
GOOGLE_SHEET_ID                 # Required
GOOGLE_SHEETS_CREDENTIALS_PATH  # Local dev: path to credentials.json
GOOGLE_SHEETS_CREDENTIALS_JSON  # Cloud: raw JSON or base64
DATA_DIR                        # Optional: persistent state directory
POLL_ENABLED                    # Default: true
POLL_INTERVAL_SECONDS           # Default: 900
POLL_LOOKBACK_HOURS             # Default: 24
```

## API Endpoints

- `GET /health` - Health check
- `GET /processed` - List processed recordings
- `DELETE /processed/{id}` - Allow reprocessing
- `POST /process/{id}` - Manual process (add `?force=true` to reprocess)
- `GET /recordings?query=name` - Search tldv recordings
- `POST /webhook/tldv` - tldv webhook receiver

## Google Sheets Structure

**Sales Tab (A-L):** Owner | Customer Name | Call Date | Next Steps | Due Date | Completed? | Recording Link | Follow-up Email | Marketing Worthy? | Main Topics (if marketing worthy) | Transcript (if marketing worthy) | Volume

**CS Tab (A-J):** Customer Name | Call Date | Next Steps | Due Date | Completed? | Recording Link | Follow-up Email | Marketing Worthy? | Main Topics | Volume

## Debugging

```bash
# Check if recording was processed
curl "https://lido-cs-automation-production.up.railway.app/processed" | grep "ID"

# Force reprocess
curl -X DELETE "https://lido-cs-automation-production.up.railway.app/processed/ID"
curl -X POST "https://lido-cs-automation-production.up.railway.app/process/ID"
```
