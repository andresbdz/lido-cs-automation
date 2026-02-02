"""Tests for tldv client."""

import pytest
from services.tldv_client import should_process_recording, TldvClient, TldvAuthenticationError


class TestShouldProcessRecording:
    """Tests for should_process_recording helper function."""

    def test_matches_customer_success(self):
        """Should match titles containing 'customer success'."""
        assert should_process_recording("Customer Success Call - Acme Corp") is True
        assert should_process_recording("Weekly customer success sync") is True
        assert should_process_recording("CUSTOMER SUCCESS review") is True

    def test_matches_check_in_variations(self):
        """Should match all variations of check-in."""
        assert should_process_recording("Weekly Check-in") is True
        assert should_process_recording("Team checkin meeting") is True
        assert should_process_recording("Monthly check in") is True
        assert should_process_recording("Check-In with client") is True
        assert should_process_recording("Q4 CHECK-IN") is True

    def test_no_match(self):
        """Should not match unrelated titles."""
        assert should_process_recording("Engineering standup") is False
        assert should_process_recording("Product roadmap review") is False
        assert should_process_recording("Sales call") is False
        assert should_process_recording("") is False
        assert should_process_recording("checking inventory") is False  # 'checking' != 'check-in'

    def test_handles_none_and_empty(self):
        """Should handle None and empty strings gracefully."""
        assert should_process_recording("") is False
        assert should_process_recording(None) is False


class TestTldvClientInit:
    """Tests for TldvClient initialization."""

    def test_requires_api_key(self, monkeypatch):
        """Should raise error when no API key provided."""
        monkeypatch.delenv("TLDV_API_KEY", raising=False)
        with pytest.raises(TldvAuthenticationError):
            TldvClient()

    def test_accepts_api_key_param(self):
        """Should accept API key as parameter."""
        client = TldvClient(api_key="test-key")
        assert client.api_key == "test-key"

    def test_reads_from_env(self, monkeypatch):
        """Should read API key from environment."""
        monkeypatch.setenv("TLDV_API_KEY", "env-test-key")
        client = TldvClient()
        assert client.api_key == "env-test-key"
