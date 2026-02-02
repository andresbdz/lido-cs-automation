"""
Google Sheets client for writing customer success data.

Uses service account authentication to append data to Google Sheets.
"""

import logging
import os
from typing import Optional

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

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


class SheetsClient:
    """
    Client for interacting with Google Sheets API.

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
        credentials_path: Optional[str] = None,
        sheet_id: Optional[str] = None,
    ):
        """
        Initialize the Sheets client.

        Args:
            credentials_path: Path to service account JSON file.
                If not provided, reads from GOOGLE_SHEETS_CREDENTIALS_PATH env var.
            sheet_id: Google Sheet ID.
                If not provided, reads from GOOGLE_SHEET_ID env var.
        """
        self.credentials_path = credentials_path or os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH")
        self.sheet_id = sheet_id or os.getenv("GOOGLE_SHEET_ID")

        if not self.credentials_path:
            raise SheetsAuthenticationError(
                "Credentials path not provided. Set GOOGLE_SHEETS_CREDENTIALS_PATH environment variable."
            )

        if not self.sheet_id:
            raise SheetsClientError(
                "Sheet ID not provided. Set GOOGLE_SHEET_ID environment variable."
            )

        if not os.path.exists(self.credentials_path):
            raise SheetsAuthenticationError(
                f"Credentials file not found: {self.credentials_path}"
            )

        self.service = self._build_service()

    def _build_service(self):
        """Build the Google Sheets API service."""
        try:
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path,
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

    def get_sheet_info(self) -> dict:
        """
        Get information about the spreadsheet.

        Returns:
            Dict with spreadsheet title and sheet names.
        """
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
