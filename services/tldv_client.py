"""
tldv.io API Client for fetching meeting recordings and transcripts.

API Documentation: https://doc.tldv.io/index.html
Base URL: https://pasta.tldv.io
"""

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# API Configuration
BASE_URL = "https://pasta.tldv.io"
API_VERSION = "v1alpha1"


class TldvAPIError(Exception):
    """Base exception for tldv API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class TldvAuthenticationError(TldvAPIError):
    """Raised when API authentication fails (401)."""
    pass


class TldvAuthorizationError(TldvAPIError):
    """Raised when API authorization fails (403)."""
    pass


class TldvNotFoundError(TldvAPIError):
    """Raised when a resource is not found (404)."""
    pass


class TldvValidationError(TldvAPIError):
    """Raised when request validation fails (400)."""
    pass


class TldvRateLimitError(TldvAPIError):
    """Raised when rate limit is exceeded (429)."""
    pass


@dataclass
class Speaker:
    """Represents a speaker in a transcript."""
    name: str
    id: Optional[str] = None


@dataclass
class TranscriptSegment:
    """Represents a segment of transcript."""
    speaker: str
    text: str
    start_time: float
    end_time: float


@dataclass
class Recording:
    """Represents a tldv recording with metadata and transcript."""
    id: str
    name: str
    happened_at: Optional[str]
    duration: Optional[int]
    organizer: Optional[dict]
    invitees: Optional[list]
    transcript: Optional[list[TranscriptSegment]] = None
    highlights: Optional[list] = None


class TldvClient:
    """
    Client for interacting with the tldv.io API.

    Usage:
        client = TldvClient()  # Uses TLDV_API_KEY from environment
        recordings = client.list_recordings(limit=10)
        recording = client.get_recording("meeting-id")
    """

    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        """
        Initialize the tldv client.

        Args:
            api_key: tldv API key. If not provided, reads from TLDV_API_KEY env var.
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key or os.getenv("TLDV_API_KEY")
        if not self.api_key:
            raise TldvAuthenticationError(
                "API key not provided. Set TLDV_API_KEY environment variable or pass api_key parameter."
            )

        self.timeout = timeout
        self.base_url = f"{BASE_URL}/{API_VERSION}"
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry logic."""
        session = requests.Session()

        # Configure retry strategy for transient errors
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            raise_on_status=False,
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        # Set default headers
        session.headers.update({
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

        return session

    def _handle_response(self, response: requests.Response) -> dict:
        """
        Handle API response and raise appropriate exceptions for errors.

        Args:
            response: The requests Response object.

        Returns:
            Parsed JSON response data.

        Raises:
            TldvAPIError: For various API error conditions.
        """
        status_code = response.status_code

        # Log request details
        logger.debug(
            f"API Response: {response.request.method} {response.request.url} -> {status_code}"
        )

        if status_code == 200:
            return response.json()

        # Parse error message from response if available
        try:
            error_data = response.json()
            error_message = error_data.get("message", response.text)
        except (ValueError, KeyError):
            error_message = response.text or f"HTTP {status_code}"

        if status_code == 400:
            logger.error(f"Validation error: {error_message}")
            raise TldvValidationError(error_message, status_code)

        if status_code == 401:
            logger.error("Authentication failed - check API key")
            raise TldvAuthenticationError(
                "Authentication failed. Please verify your API key.", status_code
            )

        if status_code == 403:
            logger.error(f"Authorization failed: {error_message}")
            raise TldvAuthorizationError(
                f"Insufficient permissions: {error_message}", status_code
            )

        if status_code == 404:
            logger.warning(f"Resource not found: {error_message}")
            raise TldvNotFoundError(
                f"Resource not found: {error_message}", status_code
            )

        if status_code == 429:
            retry_after = response.headers.get("Retry-After", "60")
            logger.warning(f"Rate limited. Retry after {retry_after}s")
            raise TldvRateLimitError(
                f"Rate limit exceeded. Retry after {retry_after} seconds.", status_code
            )

        # Generic error for other status codes
        logger.error(f"API error {status_code}: {error_message}")
        raise TldvAPIError(f"API error: {error_message}", status_code)

    def _request_with_rate_limit_handling(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict] = None,
        max_retries: int = 3,
    ) -> dict:
        """
        Make an API request with automatic rate limit handling.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: Query parameters
            max_retries: Maximum number of retries for rate limiting

        Returns:
            Parsed JSON response
        """
        url = f"{self.base_url}/{endpoint}"
        retries = 0

        while retries <= max_retries:
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    timeout=self.timeout,
                )
                return self._handle_response(response)

            except TldvRateLimitError as e:
                retries += 1
                if retries > max_retries:
                    raise

                # Extract retry delay from error message or use exponential backoff
                wait_time = min(60, 2 ** retries)
                logger.info(f"Rate limited. Waiting {wait_time}s before retry {retries}/{max_retries}")
                time.sleep(wait_time)

        raise TldvAPIError("Max retries exceeded")

    def list_recordings(
        self,
        limit: int = 50,
        page: int = 1,
        query: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        only_participated: Optional[bool] = None,
    ) -> list[dict]:
        """
        List recent recordings from tldv.

        Args:
            limit: Number of recordings to return (1-100).
            page: Page number for pagination (starts at 1).
            query: Search query to filter recordings.
            from_date: Filter recordings from this date (ISO format).
            to_date: Filter recordings until this date (ISO format).
            only_participated: Only return meetings user participated in.

        Returns:
            List of recording metadata dictionaries.
        """
        # Validate limit
        limit = max(1, min(100, limit))

        params = {
            "limit": limit,
            "page": page,
        }

        if query:
            params["query"] = query
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        if only_participated is not None:
            params["onlyParticipated"] = only_participated

        logger.info(f"Fetching recordings (limit={limit}, page={page})")

        response = self._request_with_rate_limit_handling("GET", "meetings", params)

        # Response may be paginated - extract results
        if isinstance(response, list):
            recordings = response
        else:
            recordings = response.get("results", response.get("data", []))

        logger.info(f"Retrieved {len(recordings)} recordings")
        return recordings

    def get_recording(self, recording_id: str, include_transcript: bool = True) -> Recording:
        """
        Get a recording with full details and transcript.

        Args:
            recording_id: The unique ID of the recording.
            include_transcript: Whether to fetch the transcript (default True).

        Returns:
            Recording object with metadata and optional transcript.

        Raises:
            TldvNotFoundError: If the recording doesn't exist.
        """
        logger.info(f"Fetching recording: {recording_id}")

        # Get meeting metadata
        meeting_data = self._request_with_rate_limit_handling(
            "GET", f"meetings/{recording_id}"
        )

        # Log full meeting response structure for debugging
        logger.info(f"Meeting API response keys: {list(meeting_data.keys())}")
        logger.debug(f"Full meeting response: {meeting_data}")

        recording = Recording(
            id=meeting_data.get("id", recording_id),
            name=meeting_data.get("name", ""),
            happened_at=meeting_data.get("happenedAt"),
            duration=meeting_data.get("duration"),
            organizer=meeting_data.get("organizer"),
            invitees=meeting_data.get("invitees", []),
        )

        # Fetch transcript if requested
        if include_transcript:
            segments = self._fetch_transcript(recording_id, meeting_data)
            recording.transcript = segments

        return recording

    def _fetch_transcript(self, recording_id: str, meeting_data: dict) -> Optional[list[TranscriptSegment]]:
        """
        Fetch transcript using multiple methods to handle tldv API inconsistencies.

        Args:
            recording_id: The recording ID.
            meeting_data: The meeting metadata (may contain transcript).

        Returns:
            List of TranscriptSegment or None if unavailable.
        """
        segments = []

        # Method 1: Check if transcript is embedded in meeting data
        embedded_transcript = self._extract_transcript_from_meeting(meeting_data)
        if embedded_transcript:
            logger.info(f"Found embedded transcript with {len(embedded_transcript)} segments")
            return embedded_transcript

        # Method 2: Try the dedicated transcript endpoint
        try:
            transcript_data = self._request_with_rate_limit_handling(
                "GET", f"meetings/{recording_id}/transcript"
            )

            # Log full transcript response for debugging
            logger.info(f"Transcript API response type: {type(transcript_data).__name__}")
            if isinstance(transcript_data, dict):
                logger.info(f"Transcript API response keys: {list(transcript_data.keys())}")
                logger.debug(f"Full transcript response: {transcript_data}")

            segments = self._parse_transcript_response(transcript_data)

            if segments:
                logger.info(f"Fetched transcript with {len(segments)} segments from /transcript endpoint")
                return segments

        except TldvNotFoundError:
            logger.warning(f"Transcript endpoint returned 404 for {recording_id}")

        # Method 3: Try fetching with different query parameters
        if not segments:
            segments = self._try_alternative_transcript_fetch(recording_id)

        if segments:
            return segments

        logger.warning(f"No transcript available via any method for recording {recording_id}")
        return None

    def _extract_transcript_from_meeting(self, meeting_data: dict) -> Optional[list[TranscriptSegment]]:
        """
        Extract transcript from meeting metadata if embedded.

        Checks for various field names tldv might use.
        """
        # Check common field names for embedded transcript
        transcript_fields = [
            "transcript",
            "full_transcript",
            "fullTranscript",
            "transcription",
            "text",
            "content",
            "sentences",
        ]

        for field in transcript_fields:
            if field in meeting_data:
                data = meeting_data[field]
                logger.info(f"Found transcript in meeting data field '{field}': type={type(data).__name__}")

                if isinstance(data, str) and data.strip():
                    # Plain text transcript - convert to single segment
                    logger.info(f"Converting plain text transcript ({len(data)} chars)")
                    return [TranscriptSegment(
                        speaker="Transcript",
                        text=data,
                        start_time=0,
                        end_time=0,
                    )]
                elif isinstance(data, list) and data:
                    return self._parse_transcript_response(data)
                elif isinstance(data, dict):
                    return self._parse_transcript_response(data)

        return None

    def _parse_transcript_response(self, transcript_data) -> list[TranscriptSegment]:
        """
        Parse transcript data from various response formats.

        Args:
            transcript_data: Raw transcript response (list or dict).

        Returns:
            List of TranscriptSegment objects.
        """
        segments = []

        # Handle list response (array of segments)
        if isinstance(transcript_data, list):
            for segment in transcript_data:
                if isinstance(segment, dict):
                    seg = self._parse_segment(segment)
                    if seg:
                        segments.append(seg)
                elif isinstance(segment, str):
                    # Plain text segments
                    segments.append(TranscriptSegment(
                        speaker="Speaker",
                        text=segment,
                        start_time=0,
                        end_time=0,
                    ))

        # Handle dict response
        elif isinstance(transcript_data, dict):
            # Check for nested transcript data
            nested_fields = ["sentences", "segments", "items", "data", "transcript", "results"]
            for field in nested_fields:
                if field in transcript_data and isinstance(transcript_data[field], list):
                    logger.info(f"Found transcript segments in '{field}' field")
                    return self._parse_transcript_response(transcript_data[field])

            # Check for full_transcript or text field
            text_fields = ["full_transcript", "fullTranscript", "text", "content", "transcription"]
            for field in text_fields:
                if field in transcript_data:
                    text = transcript_data[field]
                    if isinstance(text, str) and text.strip():
                        logger.info(f"Found plain text transcript in '{field}' ({len(text)} chars)")
                        return [TranscriptSegment(
                            speaker="Transcript",
                            text=text,
                            start_time=0,
                            end_time=0,
                        )]

        return segments

    def _parse_segment(self, segment: dict) -> Optional[TranscriptSegment]:
        """Parse a single transcript segment from various formats."""
        # Try different field names for text
        text = (
            segment.get("text") or
            segment.get("content") or
            segment.get("sentence") or
            segment.get("words") or
            ""
        )

        if not text:
            return None

        # Try different field names for speaker
        speaker = (
            segment.get("speaker") or
            segment.get("speakerName") or
            segment.get("speaker_name") or
            segment.get("name") or
            segment.get("participant") or
            "Unknown"
        )

        # Try different field names for timestamps
        start_time = (
            segment.get("startTime") or
            segment.get("start_time") or
            segment.get("start") or
            segment.get("from") or
            0
        )

        end_time = (
            segment.get("endTime") or
            segment.get("end_time") or
            segment.get("end") or
            segment.get("to") or
            0
        )

        return TranscriptSegment(
            speaker=str(speaker),
            text=str(text),
            start_time=float(start_time) if start_time else 0,
            end_time=float(end_time) if end_time else 0,
        )

    def _try_alternative_transcript_fetch(self, recording_id: str) -> Optional[list[TranscriptSegment]]:
        """
        Try alternative methods to fetch transcript.

        Args:
            recording_id: The recording ID.

        Returns:
            List of segments or None.
        """
        # Try with format parameter
        alternative_endpoints = [
            f"meetings/{recording_id}/transcript?format=json",
            f"meetings/{recording_id}/transcript?include=speakers",
            f"meetings/{recording_id}/transcription",
        ]

        for endpoint in alternative_endpoints:
            try:
                logger.info(f"Trying alternative endpoint: {endpoint}")
                response = self._request_with_rate_limit_handling("GET", endpoint)

                logger.info(f"Alternative endpoint response type: {type(response).__name__}")
                if isinstance(response, dict):
                    logger.info(f"Alternative endpoint response keys: {list(response.keys())}")

                segments = self._parse_transcript_response(response)
                if segments:
                    logger.info(f"Found {len(segments)} segments via {endpoint}")
                    return segments

            except (TldvNotFoundError, TldvAPIError) as e:
                logger.debug(f"Alternative endpoint {endpoint} failed: {e}")
                continue

        return None

    def get_transcript_text(self, recording_id: str) -> str:
        """
        Get the full transcript as formatted text.

        Args:
            recording_id: The unique ID of the recording.

        Returns:
            Formatted transcript string with speaker labels.
        """
        recording = self.get_recording(recording_id, include_transcript=True)

        if not recording.transcript:
            return ""

        lines = []
        current_speaker = None

        for segment in recording.transcript:
            if segment.speaker != current_speaker:
                current_speaker = segment.speaker
                lines.append(f"\n{current_speaker}:")

            lines.append(f"  {segment.text}")

        return "\n".join(lines).strip()

    def get_highlights(self, recording_id: str) -> list[dict]:
        """
        Get highlights/notes for a recording.

        Args:
            recording_id: The unique ID of the recording.

        Returns:
            List of highlight dictionaries with text, startTime, source, topic.
        """
        logger.info(f"Fetching highlights for recording: {recording_id}")

        try:
            response = self._request_with_rate_limit_handling(
                "GET", f"meetings/{recording_id}/highlights"
            )
            highlights = response if isinstance(response, list) else response.get("highlights", [])
            logger.info(f"Fetched {len(highlights)} highlights")
            return highlights
        except TldvNotFoundError:
            logger.warning(f"No highlights found for recording {recording_id}")
            return []


def should_process_recording(title: str) -> bool:
    """
    Determine if a recording should be processed based on its title.

    A recording should be processed if the title contains:
    - "customer success" (case insensitive)
    - Any variation of "check-in" (check-in, checkin, check in)

    Args:
        title: The recording/meeting title.

    Returns:
        True if the recording should be processed, False otherwise.
    """
    if not title:
        return False

    title_lower = title.lower()

    # Check for "customer success"
    if "customer success" in title_lower:
        logger.debug(f"Recording matches 'customer success': {title}")
        return True

    # Check for variations of "check-in"
    # Matches: check-in, checkin, check in, Check-In, etc.
    checkin_pattern = r"check[-\s]?in"
    if re.search(checkin_pattern, title_lower):
        logger.debug(f"Recording matches 'check-in' pattern: {title}")
        return True

    return False
