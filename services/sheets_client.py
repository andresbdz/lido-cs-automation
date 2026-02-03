"""
Google Sheets client for writing customer success data.

Uses service account authentication to append data to Google Sheets.
Supports credentials via file path OR JSON string (for cloud deployment).

Note: All Google API imports and authentication are lazy-loaded to support
cloud deployments where credentials are only available at runtime.
"""

import base64
import json
import logging
import os
from typing import TYPE_CHECKING, Optional

# Type hints only - not imported at runtime
if TYPE_CHECKING:
    from googleapiclient._apis.sheets.v4 import SheetsResource

logger = logging.getLogger(__name__)

# Google Sheets API configuration
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


class SheetsClientError(Exception):
    """Base exception for Sheets client errors."""
    pass


class SheetsAuthenticationError(SheetsClientError):
    """Raised when authentication fails."""
    pass


class SheetsAPIError(SheetsClientError):
    """Raised when API call fails."""
    pass


def _parse_credentials_json(credentials_json: str) -> dict:
    """
    Parse credentials JSON from string (raw JSON or base64 encoded).

    Args:
        credentials_json: Either raw JSON string or base64 encoded JSON.

    Returns:
        Parsed credentials dictionary.

    Raises:
        SheetsAuthenticationError: If parsing fails.
    """
    # First, try parsing as raw JSON
    try:
        return json.loads(credentials_json)
    except json.JSONDecodeError:
        pass

    # Try base64 decoding
    try:
        decoded = base64.b64decode(credentials_json).decode("utf-8")
        return json.loads(decoded)
    except Exception:
        pass

    # Try base64 with URL-safe alphabet
    try:
        decoded = base64.urlsafe_b64decode(credentials_json).decode("utf-8")
        return json.loads(decoded)
    except Exception:
        pass

    raise SheetsAuthenticationError(
        "Failed to parse GOOGLE_SHEETS_CREDENTIALS_JSON. "
        "Ensure it's valid JSON or base64-encoded JSON."
    )


class SheetsClient:
    """
    Client for interacting with Google Sheets API.

    Supports two authentication methods:
    1. GOOGLE_SHEETS_CREDENTIALS_JSON env var (raw JSON or base64 encoded)
       - Best for cloud deployments (Railway, Heroku, etc.)
    2. GOOGLE_SHEETS_CREDENTIALS_PATH env var (path to JSON file)
       - Best for local development

    All credentials and Google API initialization are lazy-loaded,
    meaning they only happen when the client is instantiated, not
    when the module is imported.

    Usage:
        client = SheetsClient()
        client.append_next_steps(
            customer_name="Acme Corp",
            call_date="2024-01-15",
            next_steps="1. Send docs\\n2. Schedule demo",
            due_date="2024-01-22"
        )
    """

    def __init__(
        self,
        credentials_json: Optional[str] = None,
        credentials_path: Optional[str] = None,
        sheet_id: Optional[str] = None,
    ):
        """
        Initialize the Sheets client.

        All credential loading and Google API initialization happens here,
        not at module import time.

        Args:
            credentials_json: Service account credentials as JSON string (raw or base64).
                If not provided, reads from GOOGLE_SHEETS_CREDENTIALS_JSON env var.
            credentials_path: Path to service account JSON file.
                If not provided, reads from GOOGLE_SHEETS_CREDENTIALS_PATH env var.
                Only used if credentials_json is not provided.
            sheet_id: Google Sheet ID.
                If not provided, reads from GOOGLE_SHEET_ID env var.
        """
        self.sheet_id = sheet_id or os.getenv("GOOGLE_SHEET_ID")

        if not self.sheet_id:
            raise SheetsClientError(
                "Sheet ID not provided. Set GOOGLE_SHEET_ID environment variable."
            )

        # Try credentials JSON first (for cloud deployment)
        creds_json = credentials_json or os.getenv("GOOGLE_SHEETS_CREDENTIALS_JSON")
        creds_path = credentials_path or os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH")

        if creds_json:
            logger.info("Using credentials from GOOGLE_SHEETS_CREDENTIALS_JSON")
            self._credentials_info = _parse_credentials_json(creds_json)
            self._auth_method = "json"
            self._credentials_path = None
        elif creds_path:
            if not os.path.exists(creds_path):
                raise SheetsAuthenticationError(
                    f"Credentials file not found: {creds_path}"
                )
            logger.info(f"Using credentials from file: {creds_path}")
            self._credentials_info = None
            self._auth_method = "file"
            self._credentials_path = creds_path
        else:
            raise SheetsAuthenticationError(
                "No credentials provided. Set either:\n"
                "  - GOOGLE_SHEETS_CREDENTIALS_JSON (for cloud deployment)\n"
                "  - GOOGLE_SHEETS_CREDENTIALS_PATH (for local development)"
            )

        self._service = None  # Lazy-loaded

    @property
    def service(self):
        """Lazy-load the Google Sheets service."""
        if self._service is None:
            self._service = self._build_service()
        return self._service

    def _build_service(self):
        """
        Build the Google Sheets API service.

        Google API libraries are imported here (lazy import) to ensure
        they're only loaded when actually needed, not at module import time.
        """
        # Lazy import Google libraries - only when service is actually needed
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
        except ImportError as e:
            raise SheetsClientError(
                f"Google API libraries not installed: {e}. "
                "Run: pip install google-api-python-client google-auth"
            )

        try:
            if self._auth_method == "json":
                credentials = service_account.Credentials.from_service_account_info(
                    self._credentials_info,
                    scopes=SCOPES,
                )
            else:
                credentials = service_account.Credentials.from_service_account_file(
                    self._credentials_path,
                    scopes=SCOPES,
                )

            service = build("sheets", "v4", credentials=credentials)
            logger.info("Google Sheets service initialized successfully")
            return service
        except Exception as e:
            logger.error(f"Failed to initialize Google Sheets service: {e}")
            raise SheetsAuthenticationError(f"Failed to authenticate: {e}")

    def append_next_steps(
        self,
        customer_name: str,
        call_date: str,
        next_steps: str,
        due_date: str,
        sheet_name: str = "Sheet1",
    ) -> dict:
        """
        Append a row with next steps data to the Google Sheet.

        Args:
            customer_name: Name of the customer/prospect.
            call_date: Date of the call (YYYY-MM-DD format).
            next_steps: MECE-formatted next steps string.
            due_date: Due date for follow-up (YYYY-MM-DD format).
            sheet_name: Name of the sheet tab (default: "Sheet1").

        Returns:
            API response dict with update details.

        Raises:
            SheetsAPIError: If the API call fails.
        """
        # Lazy import for error handling
        from googleapiclient.errors import HttpError

        logger.info(f"Appending next steps for {customer_name} (call: {call_date})")

        # Prepare row data
        # Columns: Customer Name | Call Date | Next Steps | Due Date | Completed?
        row_data = [
            customer_name,
            call_date,
            next_steps,
            due_date,
            "No",  # Default: not completed
        ]

        # Build the request
        range_name = f"{sheet_name}!A:E"
        body = {
            "values": [row_data],
        }

        try:
            result = (
                self.service.spreadsheets()
                .values()
                .append(
                    spreadsheetId=self.sheet_id,
                    range=range_name,
                    valueInputOption="USER_ENTERED",
                    insertDataOption="INSERT_ROWS",
                    body=body,
                )
                .execute()
            )

            updated_range = result.get("updates", {}).get("updatedRange", "unknown")
            logger.info(f"Successfully appended row to {updated_range}")

            return result

        except HttpError as e:
            error_message = str(e)
            logger.error(f"Google Sheets API error: {error_message}")

            if e.resp.status == 403:
                raise SheetsAuthenticationError(
                    "Permission denied. Ensure the service account has edit access to the sheet."
                )
            elif e.resp.status == 404:
                raise SheetsAPIError(
                    f"Sheet not found: {self.sheet_id}. Check the GOOGLE_SHEET_ID value."
                )
            else:
                raise SheetsAPIError(f"API error: {error_message}")

        except Exception as e:
            logger.error(f"Unexpected error appending to sheet: {e}")
            raise SheetsAPIError(f"Failed to append data: {e}")

    def append_qa_pairs(
        self,
        qa_pairs: list[dict],
        call_title: str,
        call_date: str,
        customer_name: str = "",
        sheet_name: str = "Knowledge Base",
    ) -> dict:
        """
        Append Q&A pairs to a Knowledge Base sheet tab.

        Args:
            qa_pairs: List of dicts with 'question', 'answer', 'topic' keys.
            call_title: Title of the meeting/call.
            call_date: Date of the call (YYYY-MM-DD format).
            customer_name: Customer name or email.
            sheet_name: Name of the sheet tab (default: "Knowledge Base").

        Returns:
            API response dict with update details.

        Raises:
            SheetsAPIError: If the API call fails.
        """
        from googleapiclient.errors import HttpError

        logger.info(f"Appending {len(qa_pairs)} Q&A pairs for {call_title}")

        # Ensure headers exist on the Knowledge Base tab
        headers = ["Date", "Customer", "Meeting", "Topic", "Question", "Answer"]
        header_range = f"{sheet_name}!A1:F1"
        try:
            result = (
                self.service.spreadsheets()
                .values()
                .get(spreadsheetId=self.sheet_id, range=header_range)
                .execute()
            )
            existing = result.get("values", [[]])[0] if result.get("values") else []
            if existing != headers:
                self.service.spreadsheets().values().update(
                    spreadsheetId=self.sheet_id,
                    range=header_range,
                    valueInputOption="RAW",
                    body={"values": [headers]},
                ).execute()
                logger.info("Knowledge Base headers updated")
        except HttpError as e:
            if e.resp.status == 400:
                # Tab might not exist -- that's ok, append will create it
                logger.warning(f"Could not check headers for '{sheet_name}': {e}")
            else:
                raise SheetsAPIError(f"Failed to ensure KB headers: {e}")

        # Build rows
        rows = []
        for qa in qa_pairs:
            rows.append([
                call_date,
                customer_name,
                call_title,
                qa.get("topic", ""),
                qa.get("question", ""),
                qa.get("answer", ""),
            ])

        range_name = f"{sheet_name}!A:F"
        body = {"values": rows}

        try:
            result = (
                self.service.spreadsheets()
                .values()
                .append(
                    spreadsheetId=self.sheet_id,
                    range=range_name,
                    valueInputOption="USER_ENTERED",
                    insertDataOption="INSERT_ROWS",
                    body=body,
                )
                .execute()
            )

            updated_range = result.get("updates", {}).get("updatedRange", "unknown")
            logger.info(f"Successfully appended {len(rows)} Q&A rows to {updated_range}")
            return result

        except HttpError as e:
            error_message = str(e)
            logger.error(f"Google Sheets API error (KB): {error_message}")
            if e.resp.status == 403:
                raise SheetsAuthenticationError(
                    "Permission denied. Ensure the service account has edit access."
                )
            elif e.resp.status == 404:
                raise SheetsAPIError(
                    f"Sheet not found: {self.sheet_id}. Check the GOOGLE_SHEET_ID value."
                )
            else:
                raise SheetsAPIError(f"API error: {error_message}")

        except Exception as e:
            logger.error(f"Unexpected error appending Q&A pairs: {e}")
            raise SheetsAPIError(f"Failed to append Q&A data: {e}")

    def get_sheet_info(self) -> dict:
        """
        Get information about the spreadsheet.

        Returns:
            Dict with spreadsheet title and sheet names.
        """
        from googleapiclient.errors import HttpError

        try:
            result = (
                self.service.spreadsheets()
                .get(spreadsheetId=self.sheet_id)
                .execute()
            )

            return {
                "title": result.get("properties", {}).get("title"),
                "sheets": [
                    sheet.get("properties", {}).get("title")
                    for sheet in result.get("sheets", [])
                ],
            }

        except HttpError as e:
            logger.error(f"Failed to get sheet info: {e}")
            raise SheetsAPIError(f"Failed to get sheet info: {e}")

    def ensure_headers(self, sheet_name: str = "Sheet1") -> None:
        """
        Ensure the sheet has the correct headers in row 1.

        Args:
            sheet_name: Name of the sheet tab.
        """
        from googleapiclient.errors import HttpError

        headers = [
            "Customer Name",
            "Call Date",
            "Next Steps",
            "Due Date",
            "Completed?",
        ]

        range_name = f"{sheet_name}!A1:E1"
        body = {"values": [headers]}

        try:
            # Check if headers exist
            result = (
                self.service.spreadsheets()
                .values()
                .get(spreadsheetId=self.sheet_id, range=range_name)
                .execute()
            )

            existing = result.get("values", [[]])[0] if result.get("values") else []

            if existing != headers:
                # Update headers
                self.service.spreadsheets().values().update(
                    spreadsheetId=self.sheet_id,
                    range=range_name,
                    valueInputOption="RAW",
                    body=body,
                ).execute()
                logger.info("Sheet headers updated")
            else:
                logger.debug("Sheet headers already correct")

        except HttpError as e:
            logger.error(f"Failed to ensure headers: {e}")
            raise SheetsAPIError(f"Failed to ensure headers: {e}")
