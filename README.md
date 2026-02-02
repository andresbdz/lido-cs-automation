# Lido CS Automation

Webhook-based automation for processing tldv.io meeting transcripts and updating customer success data.

## Features

- Receives webhooks from tldv.io (`MeetingReady`, `TranscriptReady` events)
- Filters recordings by title (customer success / check-in calls)
- Extracts MECE-structured next steps using Claude
- Extracts Q&A pairs for knowledge base
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

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with service status |
| `/webhook/tldv` | POST | Receives tldv.io webhook events |
| `/processed` | GET | List all processed recordings |
| `/processed/{id}` | DELETE | Remove from processed list (allow reprocess) |
| `/test` | POST | Test with a recording ID |
| `/test/mock` | POST | Test with mock data |

## Webhook Events

| Event | Behavior |
|-------|----------|
| `MeetingReady` | Acknowledged, waits for transcript |
| `TranscriptReady` | Triggers full processing pipeline |

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

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `TLDV_API_KEY` | Yes | tldv.io API key |
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key |
| `GOOGLE_SHEET_ID` | Yes | Google Sheet ID |
| `GOOGLE_SHEETS_CREDENTIALS_JSON` | For cloud | Credentials as JSON string |
| `GOOGLE_SHEETS_CREDENTIALS_PATH` | For local | Path to credentials file |

## License

Proprietary - Internal use only
