"""
Lido CS Automation - FastAPI webhook receiver for tldv.io meeting transcripts.

Processes customer success call recordings and extracts actionable insights.
"""

import json
import logging
import os
import re
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from services import (
    TldvClient,
    TldvAPIError,
    TldvAuthenticationError,
    TldvNotFoundError,
    TranscriptAnalyzer,
    TranscriptAnalyzerError,
    SheetsClient,
    SheetsClientError,
    SheetsAuthenticationError,
    append_qa_pairs,
    KnowledgeBaseError,
    should_process_recording,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Path to tracking file for processed recordings
PROCESSED_RECORDINGS_FILE = Path(__file__).parent / "processed_recordings.json"
_tracking_lock = threading.Lock()


# ============================================================================
# Duplicate Processing Tracking
# ============================================================================

def load_processed_recordings() -> dict:
    """Load processed recordings from JSON file."""
    if not PROCESSED_RECORDINGS_FILE.exists():
        return {"recordings": {}}

    try:
        with open(PROCESSED_RECORDINGS_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading processed recordings: {e}")
        return {"recordings": {}}


def save_processed_recordings(data: dict) -> None:
    """Save processed recordings to JSON file."""
    try:
        with open(PROCESSED_RECORDINGS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except IOError as e:
        logger.error(f"Error saving processed recordings: {e}")


def is_already_processed(recording_id: str) -> bool:
    """Check if a recording has already been processed."""
    with _tracking_lock:
        data = load_processed_recordings()
        return recording_id in data.get("recordings", {})


def mark_as_processed(recording_id: str, title: str, result: dict) -> None:
    """Mark a recording as processed."""
    with _tracking_lock:
        data = load_processed_recordings()
        if "recordings" not in data:
            data["recordings"] = {}

        data["recordings"][recording_id] = {
            "title": title,
            "processed_at": datetime.now().isoformat(),
            "sheets_updated": result.get("sheets_updated", False),
            "kb_updated": result.get("knowledge_base_updated", False),
            "qa_pairs_count": result.get("qa_pairs_count", 0),
        }
        save_processed_recordings(data)
        logger.info(f"Marked recording {recording_id} as processed")


def get_all_processed() -> dict:
    """Get all processed recordings."""
    with _tracking_lock:
        return load_processed_recordings()


# ============================================================================
# Helper Functions
# ============================================================================

def _mask_key(key: Optional[str]) -> str:
    """Mask API key for safe logging."""
    if not key:
        return "NOT_SET"
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}...{key[-4:]}"


def extract_customer_name(title: str, transcript_text: Optional[str] = None) -> str:
    """
    Extract customer name from meeting title or transcript.

    Attempts to parse common meeting title formats:
    - "Customer Success Check-in - Acme Corp"
    - "Acme Corp - Weekly Check-in"
    - "Check-in with Acme Corp"
    - "Acme Corp Customer Success Call"

    Args:
        title: Meeting title.
        transcript_text: Optional transcript to search for company mentions.

    Returns:
        Extracted customer name or "Unknown Customer" if not found.
    """
    if not title:
        return "Unknown Customer"

    # Common patterns to extract customer name
    patterns = [
        # "Something - Company Name"
        r"(?:check-?in|customer success|cs call|call|meeting|sync)\s*[-–—]\s*(.+?)(?:\s*[-–—]|$)",
        # "Company Name - Something"
        r"^(.+?)\s*[-–—]\s*(?:check-?in|customer success|cs call|weekly|monthly|quarterly)",
        # "Check-in with Company Name"
        r"(?:check-?in|call|meeting|sync)\s+(?:with|for)\s+(.+?)(?:\s*[-–—]|$)",
        # "Company Name Check-in" or "Company Name Customer Success"
        r"^(.+?)\s+(?:check-?in|customer success|cs\s)",
    ]

    for pattern in patterns:
        match = re.search(pattern, title, re.IGNORECASE)
        if match:
            customer = match.group(1).strip()
            # Clean up common suffixes
            customer = re.sub(r"\s*(call|meeting|sync|weekly|monthly)$", "", customer, flags=re.IGNORECASE)
            if customer and len(customer) > 2:
                return customer

    # Fallback: use the full title if it's short enough
    if len(title) <= 50:
        return title

    return "Unknown Customer"


# Global service instances (initialized on startup)
tldv_client: Optional[TldvClient] = None
transcript_analyzer: Optional[TranscriptAnalyzer] = None
# SheetsClient is initialized lazily when needed
_sheets_client: Optional[SheetsClient] = None
_sheets_client_initialized: bool = False


def get_sheets_client() -> Optional[SheetsClient]:
    """
    Get or lazily initialize the SheetsClient.

    This allows the app to start without Google Sheets credentials
    and only fail when Sheets functionality is actually needed.
    """
    global _sheets_client, _sheets_client_initialized

    if _sheets_client_initialized:
        return _sheets_client

    _sheets_client_initialized = True

    try:
        _sheets_client = SheetsClient()
        logger.info("SheetsClient initialized successfully (lazy)")
        return _sheets_client
    except SheetsAuthenticationError as e:
        logger.warning(f"SheetsClient not available (credentials missing): {e}")
        return None
    except SheetsClientError as e:
        logger.warning(f"SheetsClient not available (config missing): {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize SheetsClient: {e}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    global tldv_client, transcript_analyzer

    logger.info("Initializing services...")

    # Log masked API keys for debugging (never log full keys)
    logger.info(f"TLDV_API_KEY: {_mask_key(os.getenv('TLDV_API_KEY'))}")
    logger.info(f"ANTHROPIC_API_KEY: {_mask_key(os.getenv('ANTHROPIC_API_KEY'))}")
    logger.info(f"GOOGLE_SHEET_ID: {os.getenv('GOOGLE_SHEET_ID', 'NOT_SET')[:20] if os.getenv('GOOGLE_SHEET_ID') else 'NOT_SET'}...")

    # Log which credentials method will be used (without accessing the actual values)
    if os.getenv("GOOGLE_SHEETS_CREDENTIALS_JSON"):
        logger.info("GOOGLE_SHEETS_CREDENTIALS_JSON: configured (will use JSON credentials)")
    elif os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH"):
        logger.info(f"GOOGLE_SHEETS_CREDENTIALS_PATH: {os.getenv('GOOGLE_SHEETS_CREDENTIALS_PATH')}")
    else:
        logger.info("Google Sheets credentials: NOT_SET (Sheets integration disabled)")

    # Initialize tldv client
    try:
        tldv_client = TldvClient()
        logger.info("TldvClient initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize TldvClient: {e}")
        tldv_client = None

    # Initialize transcript analyzer
    try:
        transcript_analyzer = TranscriptAnalyzer()
        logger.info("TranscriptAnalyzer initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize TranscriptAnalyzer: {e}")
        transcript_analyzer = None

    # Note: SheetsClient is initialized lazily when first needed
    # This allows the app to start without Google Sheets credentials
    logger.info("SheetsClient will be initialized lazily when needed")

    # Log processed recordings count
    processed = load_processed_recordings()
    logger.info(f"Loaded {len(processed.get('recordings', {}))} previously processed recordings")

    yield

    logger.info("Shutting down services...")


app = FastAPI(
    title="Lido CS Automation",
    description="Webhook receiver for tldv.io meeting transcripts",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================================
# Pydantic Models
# ============================================================================

class TldvWebhookPayload(BaseModel):
    """
    tldv.io webhook payload structure.

    MeetingReady payload:
    {
        "id": "webhook-id",
        "event": "MeetingReady",
        "data": { "id": "meeting-id", "name": "...", ... },
        "executedAt": "2024-01-15T10:00:00Z"
    }

    TranscriptReady payload:
    {
        "id": "webhook-id",
        "event": "TranscriptReady",
        "data": { "id": "transcript-id", "meetingId": "meeting-id", "data": [...] },
        "executedAt": "2024-01-15T10:00:00Z"
    }
    """
    id: Optional[str] = None  # Webhook ID
    event: str  # "MeetingReady" or "TranscriptReady"
    data: dict  # Contains meeting or transcript data
    executedAt: Optional[str] = Field(None, alias="executedAt")

    class Config:
        populate_by_name = True


class ProcessingResult(BaseModel):
    """Result of processing a recording."""
    recording_id: str
    title: str
    customer_name: Optional[str] = None
    processed: bool
    next_steps: Optional[dict] = None
    qa_pairs_count: int = 0
    sheets_updated: bool = False
    knowledge_base_updated: bool = False
    error: Optional[str] = None


class TestPayload(BaseModel):
    """Payload for test endpoint."""
    recording_id: str


class ProcessedRecording(BaseModel):
    """Information about a processed recording."""
    recording_id: str
    title: str
    processed_at: str
    sheets_updated: bool
    kb_updated: bool
    qa_pairs_count: int


# ============================================================================
# Processing Logic
# ============================================================================

def process_recording(recording_id: str) -> ProcessingResult:
    """
    Process a single recording: fetch, analyze, and store insights.

    Pipeline:
    1. Fetch recording from tldv
    2. Check if title matches processing criteria
    3. Extract next steps and Q&A pairs using Claude
    4. Write next steps to Google Sheets
    5. Append Q&A pairs to knowledge base

    Args:
        recording_id: The tldv recording/meeting ID.

    Returns:
        ProcessingResult with extracted data or error information.
    """
    logger.info(f"Processing recording: {recording_id}")

    # Check service availability
    if not tldv_client:
        logger.error("TldvClient not initialized")
        return ProcessingResult(
            recording_id=recording_id,
            title="Unknown",
            processed=False,
            error="TldvClient not initialized - check TLDV_API_KEY",
        )

    if not transcript_analyzer:
        logger.error("TranscriptAnalyzer not initialized")
        return ProcessingResult(
            recording_id=recording_id,
            title="Unknown",
            processed=False,
            error="TranscriptAnalyzer not initialized - check ANTHROPIC_API_KEY",
        )

    # Fetch recording from tldv
    try:
        logger.info(f"Fetching recording from tldv: {recording_id}")
        recording = tldv_client.get_recording(recording_id)
        logger.info(f"Recording fetched: '{recording.name}' (duration: {recording.duration}s)")
    except TldvNotFoundError:
        logger.warning(f"Recording not found: {recording_id}")
        return ProcessingResult(
            recording_id=recording_id,
            title="Unknown",
            processed=False,
            error=f"Recording not found: {recording_id}",
        )
    except TldvAuthenticationError as e:
        logger.error(f"tldv authentication failed: {e}")
        return ProcessingResult(
            recording_id=recording_id,
            title="Unknown",
            processed=False,
            error="tldv authentication failed - check API key",
        )
    except TldvAPIError as e:
        logger.error(f"tldv API error: {e}")
        return ProcessingResult(
            recording_id=recording_id,
            title="Unknown",
            processed=False,
            error=f"tldv API error: {e.message}",
        )

    # Extract customer name: use first non-@trylido.com attendee email, fall back to title
    customer_name = None
    if recording.invitees:
        for invitee in recording.invitees:
            email = invitee.get("email", "") if isinstance(invitee, dict) else ""
            if email and not email.endswith("@trylido.com"):
                customer_name = email
                break
    if not customer_name:
        customer_name = extract_customer_name(recording.name)
    logger.info(f"Extracted customer name: {customer_name}")

    # Check if recording should be processed
    if not should_process_recording(recording.name):
        logger.info(f"Skipping recording (title doesn't match criteria): '{recording.name}'")
        return ProcessingResult(
            recording_id=recording_id,
            title=recording.name,
            customer_name=customer_name,
            processed=False,
            error="Recording title doesn't match processing criteria",
        )

    # Check if transcript is available
    if not recording.transcript:
        logger.warning(f"No transcript available for recording: {recording_id}")
        return ProcessingResult(
            recording_id=recording_id,
            title=recording.name,
            customer_name=customer_name,
            processed=False,
            error="No transcript available",
        )

    # Format transcript for analysis
    transcript_text = "\n".join(
        f"{seg.speaker}: {seg.text}" for seg in recording.transcript
    )
    logger.info(f"Transcript length: {len(transcript_text)} characters")

    # Determine call date
    call_date = recording.happened_at or datetime.now().isoformat()[:10]
    # Normalize to YYYY-MM-DD format
    call_date = call_date[:10] if len(call_date) >= 10 else call_date

    # Extract next steps
    next_steps = None
    try:
        logger.info("Extracting next steps...")
        next_steps = transcript_analyzer.extract_next_steps(transcript_text, call_date)
        logger.info(f"Next steps extracted:")
        logger.info(f"  - Prospect email: {next_steps.get('prospect_email', 'N/A')}")
        logger.info(f"  - Due date: {next_steps.get('due_date', 'N/A')}")
        logger.info(f"  - Next steps:\n{next_steps.get('next_steps', 'N/A')}")
    except TranscriptAnalyzerError as e:
        logger.error(f"Failed to extract next steps: {e}")

    # Extract Q&A pairs
    qa_pairs = []
    try:
        logger.info("Extracting Q&A pairs...")
        qa_pairs = transcript_analyzer.extract_qa_pairs(transcript_text)
        logger.info(f"Extracted {len(qa_pairs)} Q&A pairs:")
        for i, qa in enumerate(qa_pairs, 1):
            logger.info(f"  {i}. [{qa['topic']}] Q: {qa['question'][:50]}...")
    except TranscriptAnalyzerError as e:
        logger.error(f"Failed to extract Q&A pairs: {e}")

    # Write to Google Sheets
    sheets_updated = False
    if next_steps:
        client = get_sheets_client()
        if client:
            try:
                logger.info("Writing next steps to Google Sheets...")
                client.append_next_steps(
                    customer_name=customer_name,
                    call_date=call_date,
                    next_steps=next_steps.get("next_steps", ""),
                    due_date=next_steps.get("due_date", ""),
                )
                sheets_updated = True
                logger.info("Successfully updated Google Sheets")
            except SheetsClientError as e:
                logger.error(f"Failed to update Google Sheets: {e}")
        else:
            logger.warning("SheetsClient not available - skipping Sheets update")

    # Write to knowledge base
    kb_updated = False
    if qa_pairs:
        try:
            logger.info("Writing Q&A pairs to knowledge base...")
            append_qa_pairs(
                qa_pairs=qa_pairs,
                call_title=recording.name,
                call_date=call_date,
            )
            kb_updated = True
            logger.info("Successfully updated knowledge base")
        except KnowledgeBaseError as e:
            logger.error(f"Failed to update knowledge base: {e}")

    logger.info(f"Successfully processed recording: {recording_id}")

    return ProcessingResult(
        recording_id=recording_id,
        title=recording.name,
        customer_name=customer_name,
        processed=True,
        next_steps=next_steps,
        qa_pairs_count=len(qa_pairs),
        sheets_updated=sheets_updated,
        knowledge_base_updated=kb_updated,
    )


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring.

    Returns service status and readiness.
    """
    processed = load_processed_recordings()
    # Check sheets client status without initializing it if not yet used
    sheets_status = "ready" if _sheets_client else ("not_configured" if _sheets_client_initialized else "pending_initialization")
    return {
        "status": "healthy",
        "services": {
            "tldv_client": "ready" if tldv_client else "not_initialized",
            "transcript_analyzer": "ready" if transcript_analyzer else "not_initialized",
            "sheets_client": sheets_status,
        },
        "processed_recordings_count": len(processed.get("recordings", {})),
    }


@app.get("/debug/sheets")
async def debug_sheets():
    """Test Google Sheets client initialization."""
    # List all GOOGLE* env vars the app can see
    google_vars = {k: f"{len(v)} chars" if len(v) > 50 else v for k, v in os.environ.items() if "GOOGLE" in k.upper()}

    creds_json = os.getenv("GOOGLE_SHEETS_CREDENTIALS_JSON")
    creds_path = os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH")
    sheet_id = os.getenv("GOOGLE_SHEET_ID")

    env_check = {
        "GOOGLE_SHEETS_CREDENTIALS_JSON": f"{len(creds_json)} chars, starts with: {creds_json[:30]}..." if creds_json else "NOT SET",
        "GOOGLE_SHEETS_CREDENTIALS_PATH": creds_path or "NOT SET",
        "GOOGLE_SHEET_ID": sheet_id or "NOT SET",
        "all_google_vars": google_vars,
    }

    try:
        client = SheetsClient()
        return {"success": True, "message": "SheetsClient initialized", "env": env_check}
    except SheetsAuthenticationError as e:
        return {"success": False, "error_type": "auth", "error": str(e), "env": env_check}
    except SheetsClientError as e:
        return {"success": False, "error_type": "config", "error": str(e), "env": env_check}
    except Exception as e:
        return {"success": False, "error_type": "unknown", "error": f"{type(e).__name__}: {e}", "env": env_check}


@app.get("/debug/anthropic")
async def debug_anthropic():
    """Test Anthropic API connectivity."""
    import httpx

    results = {"api_reachable": False, "dns_resolved": False, "error": None}

    # Test DNS resolution
    try:
        import socket
        ip = socket.gethostbyname("api.anthropic.com")
        results["dns_resolved"] = True
        results["resolved_ip"] = ip
    except Exception as e:
        results["error"] = f"DNS failed: {e}"
        return results

    # Test HTTPS connection
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get("https://api.anthropic.com/v1/messages")
            results["api_reachable"] = True
            results["status_code"] = resp.status_code
    except Exception as e:
        results["error"] = f"Connection failed: {type(e).__name__}: {e}"

    return results


@app.get("/debug/anthropic-sdk")
async def debug_anthropic_sdk():
    """Test Anthropic SDK with a minimal API call."""
    import anthropic
    import httpx

    results = {}

    # Test 1: Raw httpx POST (sync, like SDK uses)
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": os.getenv("ANTHROPIC_API_KEY"),
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-5-20250929",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "Say ok"}],
                },
            )
            results["httpx_sync"] = {"success": True, "status": resp.status_code}
            if resp.status_code == 200:
                results["httpx_sync"]["response"] = resp.json().get("content", [{}])[0].get("text", "")
            else:
                results["httpx_sync"]["body"] = resp.text[:200]
    except Exception as e:
        results["httpx_sync"] = {"success": False, "error": f"{type(e).__name__}: {e}"}

    # Test 2: Anthropic SDK
    try:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=10,
            messages=[{"role": "user", "content": "Say 'ok'"}],
        )
        results["sdk"] = {"success": True, "response": response.content[0].text}
    except Exception as e:
        results["sdk"] = {"success": False, "error": f"{type(e).__name__}: {e}"}

    return results


@app.get("/processed")
async def get_processed_recordings():
    """
    Get list of all processed recordings.

    Use this for debugging to see what has been processed.
    """
    data = get_all_processed()
    recordings = data.get("recordings", {})

    return {
        "total": len(recordings),
        "recordings": [
            {
                "recording_id": rid,
                **info
            }
            for rid, info in recordings.items()
        ]
    }


@app.delete("/processed/{recording_id}")
async def remove_processed_recording(recording_id: str):
    """
    Remove a recording from the processed list (allows reprocessing).
    """
    with _tracking_lock:
        data = load_processed_recordings()
        if recording_id in data.get("recordings", {}):
            del data["recordings"][recording_id]
            save_processed_recordings(data)
            logger.info(f"Removed {recording_id} from processed list")
            return {"status": "removed", "recording_id": recording_id}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Recording {recording_id} not found in processed list"
            )


@app.post("/webhook/tldv")
async def tldv_webhook(request: Request):
    """
    Receive webhook events from tldv.io.

    Supported events:
    - MeetingReady: Acknowledged but not processed (no transcript yet)
    - TranscriptReady: Triggers full processing pipeline

    Payload structure:
    {
        "id": "webhook-id",
        "event": "MeetingReady" | "TranscriptReady",
        "data": { ... },
        "executedAt": "ISO timestamp"
    }
    """
    # Parse payload
    try:
        raw_payload = await request.json()
    except Exception:
        logger.error("Invalid JSON payload received")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON payload",
        )

    # Log the raw payload structure for debugging
    event_type = raw_payload.get("event", "unknown")
    webhook_id = raw_payload.get("id", "unknown")
    logger.info(f"Received tldv webhook: event={event_type}, webhook_id={webhook_id}")
    logger.debug(f"Full webhook payload: {json.dumps(raw_payload, indent=2)}")

    # Validate payload structure
    try:
        payload = TldvWebhookPayload(**raw_payload)
    except Exception as e:
        logger.error(f"Invalid webhook payload structure: {e}")
        logger.error(f"Payload keys: {list(raw_payload.keys())}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid payload structure: {e}",
        )

    # Handle MeetingReady event - acknowledge but don't process
    if payload.event == "MeetingReady":
        meeting_id = payload.data.get("id")
        meeting_name = payload.data.get("name", "Unknown")
        logger.info(f"MeetingReady event received for '{meeting_name}' (id: {meeting_id})")
        logger.info("Waiting for TranscriptReady event before processing")

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "acknowledged",
                "event": "MeetingReady",
                "meeting_id": meeting_id,
                "message": "Waiting for TranscriptReady event",
            }
        )

    # Handle TranscriptReady event - this triggers processing
    if payload.event == "TranscriptReady":
        # Extract meeting ID from TranscriptReady payload
        # Structure: data.meetingId contains the meeting ID
        meeting_id = payload.data.get("meetingId") or payload.data.get("id")

        if not meeting_id:
            logger.error(f"No meetingId in TranscriptReady payload. Data keys: {list(payload.data.keys())}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing meetingId in TranscriptReady payload",
            )

        logger.info(f"TranscriptReady event received for meeting: {meeting_id}")

        # Check for duplicate processing
        if is_already_processed(meeting_id):
            logger.info(f"Recording {meeting_id} already processed - skipping")
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "status": "already_processed",
                    "recording_id": meeting_id,
                    "message": "This recording has already been processed",
                }
            )

        # Process the recording
        result = process_recording(meeting_id)

        # Mark as processed if successful
        if result.processed:
            mark_as_processed(
                recording_id=meeting_id,
                title=result.title,
                result=result.model_dump(),
            )

        # Return appropriate status code based on result
        if not result.processed:
            if "not found" in (result.error or "").lower():
                return JSONResponse(
                    status_code=status.HTTP_404_NOT_FOUND,
                    content=result.model_dump(),
                )
            elif "authentication" in (result.error or "").lower():
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content=result.model_dump(),
                )
            # Return 200 for skipped recordings (not an error, just filtered out)
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content=result.model_dump(),
            )

        return result

    # Unknown event type
    logger.warning(f"Unknown webhook event type: {payload.event}")
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "ignored",
            "event": payload.event,
            "message": f"Unknown event type: {payload.event}",
        }
    )


@app.post("/test", response_model=ProcessingResult)
async def test_endpoint(payload: TestPayload):
    """
    Test endpoint to simulate webhook processing.

    Bypasses duplicate check for testing purposes.

    curl -X POST http://localhost:8000/test \\
        -H "Content-Type: application/json" \\
        -d '{"recording_id": "your-recording-id"}'
    """
    logger.info(f"Test endpoint called with recording_id: {payload.recording_id}")

    # Check if already processed (but still process for testing)
    if is_already_processed(payload.recording_id):
        logger.info(f"Note: Recording {payload.recording_id} was previously processed")

    result = process_recording(payload.recording_id)

    # Mark as processed if successful
    if result.processed:
        mark_as_processed(
            recording_id=payload.recording_id,
            title=result.title,
            result=result.model_dump(),
        )

    return result


@app.post("/test/mock")
async def test_mock_endpoint():
    """
    Test endpoint with mock data (no actual API calls).

    Tests the full pipeline including Sheets and knowledge base writing.
    """
    logger.info("Mock test endpoint called")

    # Mock data
    customer_name = "Acme Corp"
    call_date = "2024-01-15"
    call_title = "Customer Success Check-in - Acme Corp"

    mock_next_steps = {
        "prospect_email": "john@acme.com",
        "call_date": call_date,
        "next_steps": (
            "1. [Lido Team] Send API documentation - by EOD Friday\n"
            "2. [Prospect] Share current spreadsheet - by Monday\n"
            "3. [Lido Team] Schedule technical demo - within 2 weeks"
        ),
        "due_date": "2024-01-22",
    }

    mock_qa_pairs = [
        {
            "question": "How do I connect Google Sheets to Lido?",
            "answer": "Go to Integrations > Google Sheets > Connect, then authorize your Google account.",
            "topic": "Integrations",
        },
        {
            "question": "Can I use formulas from Google Sheets in Lido?",
            "answer": "Yes, Lido supports most Google Sheets formulas natively, plus additional Lido-specific functions.",
            "topic": "Formulas",
        },
    ]

    # Test Sheets integration
    sheets_updated = False
    client = get_sheets_client()
    if client:
        try:
            logger.info("Testing Google Sheets integration...")
            client.append_next_steps(
                customer_name=customer_name,
                call_date=call_date,
                next_steps=mock_next_steps["next_steps"],
                due_date=mock_next_steps["due_date"],
            )
            sheets_updated = True
            logger.info("Successfully wrote to Google Sheets")
        except SheetsClientError as e:
            logger.error(f"Sheets write failed: {e}")
    else:
        logger.warning("SheetsClient not available - skipping Sheets test")

    # Test knowledge base integration
    kb_updated = False
    try:
        logger.info("Testing knowledge base integration...")
        append_qa_pairs(
            qa_pairs=mock_qa_pairs,
            call_title=call_title,
            call_date=call_date,
        )
        kb_updated = True
        logger.info("Successfully wrote to knowledge base")
    except KnowledgeBaseError as e:
        logger.error(f"Knowledge base write failed: {e}")

    mock_result = {
        "recording_id": "mock-recording-123",
        "title": call_title,
        "customer_name": customer_name,
        "processed": True,
        "next_steps": mock_next_steps,
        "qa_pairs_count": len(mock_qa_pairs),
        "sheets_updated": sheets_updated,
        "knowledge_base_updated": kb_updated,
        "error": None,
    }

    logger.info("Mock processing result:")
    logger.info(f"  - Customer: {customer_name}")
    logger.info(f"  - Sheets updated: {sheets_updated}")
    logger.info(f"  - Knowledge base updated: {kb_updated}")

    return mock_result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
