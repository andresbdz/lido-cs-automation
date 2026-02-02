# Lido CS Automation

Webhook-based automation for processing tldv.io meeting transcripts and updating customer success data.

## Features

- Receives webhooks from tldv.io when meetings are recorded
- Processes transcripts using Anthropic's Claude API
- Updates Google Sheets with extracted insights

## Prerequisites

- Python 3.10+
- tldv.io account with API access
- Google Cloud project with Sheets API enabled
- Anthropic API key

## Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
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

   Copy `.env.example` to `.env` (or edit `.env` directly) and fill in your credentials:
   ```
   TLDV_API_KEY=your_tldv_api_key_here
   GOOGLE_SHEET_ID=your_google_sheet_id_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

5. **Set up Google Sheets API credentials**

   - Go to Google Cloud Console
   - Enable the Google Sheets API
   - Create a service account and download the credentials JSON
   - Place the credentials file in the project root as `credentials.json`

## Running the Application

**Development:**
```bash
uvicorn app:app --reload --port 8000
```

**Production:**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/webhook/tldv` | POST | Receives tldv.io webhook events |

## Webhook Setup

Configure your tldv.io webhook to point to:
```
https://your-domain.com/webhook/tldv
```

## Project Structure

```
lido-cs-automation/
├── app.py              # FastAPI application
├── services/           # Service modules
│   └── __init__.py
├── knowledge_base.md   # CS knowledge base
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (not in git)
├── .gitignore
└── README.md
```

## License

Proprietary - Internal use only
