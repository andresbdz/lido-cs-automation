# Lido CS Automation

Webhook-based automation for processing tldv.io meeting transcripts and updating customer success data.

**Deployed URL:** https://lido-cs-automation-production.up.railway.app

## Features

- Receives webhooks from tldv.io (`MeetingReady`, `TranscriptReady` events)
- Polling fallback (every 15 minutes) for missed webhooks
- Filters recordings by title (customer success / check-in / sales calls)
- Extracts MECE-structured next steps using Claude
- Extracts Q&A pairs for knowledge base
- Generates follow-up emails for sales calls
- Updates Google Sheets with next steps
- Appends Q&A to knowledge base markdown
- Duplicate processing prevention

## Prerequisites

- Python 3.10+
- tldv.io account with API access
- Google Cloud project with Sheets API enabled
- Anthropic API key

## Local Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/andresbdz/lido-cs-automation.git
   cd lido-cs-automation
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   Create a `.env` file:
   ```
   TLDV_API_KEY=your_tldv_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   GOOGLE_SHEET_ID=your_google_sheet_id
   GOOGLE_SHEETS_CREDENTIALS_PATH=credentials.json
   ```

5. **Set up Google Sheets API credentials**
   - Go to Google Cloud Console
   - Enable the Google Sheets API
   - Create a service account and download the credentials JSON
   - Save as `credentials.json` in project root
   - Share your Google Sheet with the service account email

6. **Run the application**
   ```bash
   uvicorn app:app --reload --port 8000
   ```

## Railway Deployment

1. **Create a new Railway project** and connect your GitHub repo

2. **Set environment variables** in Railway dashboard:

   | Variable | Value |
   |----------|-------|
   | `TLDV_API_KEY` | Your tldv API key |
   | `ANTHROPIC_API_KEY` | Your Anthropic API key |
   | `GOOGLE_SHEET_ID` | Your Google Sheet ID |
   | `GOOGLE_SHEETS_CREDENTIALS_JSON` | Full JSON content (see below) |

3. **For Google credentials**, copy the entire contents of your `credentials.json` file and paste it as the value for `GOOGLE_SHEETS_CREDENTIALS_JSON`. Railway will handle the JSON string.

   Alternatively, you can base64 encode it:
   ```bash
   base64 -w 0 credentials.json
   ```

4. **Configure the start command** in Railway:
   ```
   uvicorn app:app --host 0.0.0.0 --port $PORT
   ```

5. **Set up tldv webhook** to point to your Railway URL:
   ```
   https://your-app.railway.app/webhook/tldv
   ```

## How It Works

### Two Methods of Triggering Processing

1. **Webhooks** (primary) - tldv sends events to `/webhook/tldv` when recordings are ready
2. **Polling** (fallback) - Every 15 minutes, fetches recordings from the last 24 hours

### Processing Pipeline

1. Recording ID received (via webhook or polling)
2. Atomically claim the recording (prevents duplicates)
3. Fetch recording + transcript from tldv API
4. Classify recording based on title:
   - `"cs"` → Customer Success tab
   - `"sales"` → Sales tab (default)
   - `"skip"` → Not processed
5. Extract insights using Claude AI:
   - Next steps + due dates
   - Q&A pairs for knowledge base
   - Follow-up email (sales only)
   - Marketing worthiness (sales only)
   - Volume info (sales only)
6. Write to Google Sheets
7. Append Q&A to knowledge base
8. Mark as processed in tracking file

### Classification Logic

Recordings are classified by title (`services/tldv_client.py`):

| Classification | Title Contains | Destination |
|----------------|----------------|-------------|
| `cs` | "customer success", "check-in", "weekly", "monthly" | Customer Success tab |
| `skip` | "internal", "standup", "stand-up" | Not processed |
| `sales` | Everything else | Sales tab |

### Webhook Events

| Event | Behavior |
|-------|----------|
| `MeetingReady` | Acknowledged, waits for transcript |
| `TranscriptReady` | Triggers full processing pipeline |

## API Endpoints

Base URL: `https://lido-cs-automation-production.up.railway.app`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with service status |
| `/webhook/tldv` | POST | Receives tldv.io webhook events |
| `/processed` | GET | List all processed recordings |
| `/processed/{id}` | DELETE | Remove from processed list (allow reprocess) |
| `/process/{id}` | POST | Manually process a recording |
| `/process/{id}?force=true` | POST | Force reprocess even if already processed |
| `/recordings` | GET | List recent recordings from tldv |
| `/recordings?query=name` | GET | Search tldv recordings by name |

## Debugging

### Important: Local vs Server State

The `processed_recordings.json` file on your local machine is **NOT** the same as the one on the Railway server. When debugging:

- **Always check the server's state**, not local files
- Use the API endpoints below to query the deployed server

### Common Debugging Scenarios

#### Recording not in Google Sheet

1. **Check if it was processed on the server:**
   ```bash
   curl "https://lido-cs-automation-production.up.railway.app/processed" | grep "RECORDING_ID"
   ```

2. **If processed:** Check the correct tab (Sales vs Customer Success) based on title classification

3. **If not processed:** Check why:
   - Title might match "skip" criteria (internal/standup)
   - Transcript might not be ready yet
   - Recording might be from a different tldv user (API key access issue)

#### Recording from a new tldv user

The tldv API key is tied to a specific user. Recordings from other users may not be accessible unless:
- They're in the same tldv workspace
- The recording is shared with the API key owner
- Webhooks are configured on the new user's account

#### Polling not picking up recordings

Polling runs every 15 minutes and looks back 24 hours. Check:
- Is the server running? (`/health` endpoint)
- Is the recording within the 24-hour window?
- Are there more than 50 recordings in 24 hours? (pagination limit)

### Quick Reference Commands

```bash
# Check health
curl "https://lido-cs-automation-production.up.railway.app/health"

# Check if recording was processed
curl "https://lido-cs-automation-production.up.railway.app/processed" | grep "RECORDING_ID"

# Get full processed recordings list
curl "https://lido-cs-automation-production.up.railway.app/processed"

# List recent recordings from tldv
curl "https://lido-cs-automation-production.up.railway.app/recordings"

# Search for a specific recording
curl "https://lido-cs-automation-production.up.railway.app/recordings?query=Max"

# Manually trigger processing
curl -X POST "https://lido-cs-automation-production.up.railway.app/process/RECORDING_ID"

# Force reprocess
curl -X POST "https://lido-cs-automation-production.up.railway.app/process/RECORDING_ID?force=true"

# Remove from processed list (allows reprocessing)
curl -X DELETE "https://lido-cs-automation-production.up.railway.app/processed/RECORDING_ID"
```

### Extracting Recording ID from tldv URL

tldv URL format: `https://tldv.io/app/meetings/RECORDING_ID`

Example: `https://tldv.io/app/meetings/69a07c25e344ac00135c90a7`
Recording ID: `69a07c25e344ac00135c90a7`

## Google Sheets Structure

### Sales Tab (Columns A-K)

| A | B | C | D | E | F | G | H | I | J | K |
|---|---|---|---|---|---|---|---|---|---|---|
| Owner | Customer Name | Call Date | Next Steps | Due Date | Completed? | Recording Link | Follow-up Email | Marketing Worthy? | Main Topics | Volume |

### Customer Success Tab (Columns A-J)

| A | B | C | D | E | F | G | H | I | J |
|---|---|---|---|---|---|---|---|---|---|
| Customer Name | Call Date | Next Steps | Due Date | Completed? | Recording Link | Follow-up Email | Marketing Worthy? | Main Topics | Volume |

Note: CS tab doesn't have the "Owner" column.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `TLDV_API_KEY` | Yes | tldv.io API authentication |
| `ANTHROPIC_API_KEY` | Yes | Claude AI access |
| `GOOGLE_SHEET_ID` | Yes | Target Google Sheet ID |
| `GOOGLE_SHEETS_CREDENTIALS_JSON` | For cloud | Google auth (base64 or raw JSON) |
| `GOOGLE_SHEETS_CREDENTIALS_PATH` | For local | Path to credentials file |
| `POLL_ENABLED` | No | Enable polling (default: true) |
| `POLL_INTERVAL_SECONDS` | No | Polling frequency (default: 900 = 15 min) |
| `POLL_LOOKBACK_HOURS` | No | How far back to look (default: 24) |

## Project Structure

```
lido-cs-automation/
├── app.py                      # FastAPI application
├── services/
│   ├── __init__.py
│   ├── tldv_client.py          # tldv API client
│   ├── transcript_analyzer.py  # Claude transcript analysis
│   ├── sheets_client.py        # Google Sheets integration
│   └── knowledge_base_writer.py
├── tests/
│   ├── test_tldv_client.py
│   └── test_transcript_analyzer.py
├── knowledge_base.md           # Q&A knowledge base
├── requirements.txt
├── .env                        # Local env vars (not in git)
├── .gitignore
└── README.md
```

## Key Files

| File | Purpose |
|------|---------|
| `app.py` | Main FastAPI app, webhook handlers, processing orchestration |
| `services/tldv_client.py` | tldv API client, fetches recordings/transcripts |
| `services/transcript_analyzer.py` | Claude AI analysis |
| `services/sheets_client.py` | Google Sheets integration |
| `services/knowledge_base_writer.py` | Local markdown KB writer |
| `processed_recordings.json` | **SERVER-SIDE** tracking of processed recordings |

## License

Proprietary - Internal use only
