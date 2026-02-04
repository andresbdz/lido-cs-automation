"""Tests for tldv client."""

import pytest
from services.tldv_client import classify_recording, should_process_recording, TldvClient, TldvAuthenticationError


class TestClassifyRecording:
    """Tests for classify_recording helper function."""

    def test_cs_customer_success(self):
        """Should classify titles containing 'customer success' as CS."""
        assert classify_recording("Customer Success Call - Acme Corp") == "cs"
        assert classify_recording("Weekly customer success sync") == "cs"
        assert classify_recording("CUSTOMER SUCCESS review") == "cs"

    def test_cs_check_in_variations(self):
        """Should classify all variations of check-in as CS."""
        assert classify_recording("Weekly Check-in") == "cs"
        assert classify_recording("Team checkin meeting") == "cs"
        assert classify_recording("Monthly check in") == "cs"
        assert classify_recording("Check-In with client") == "cs"
        assert classify_recording("Q4 CHECK-IN") == "cs"

    def test_cs_weekly(self):
        """Should classify titles containing 'weekly' as CS."""
        assert classify_recording("Disney Trucking and Lido Weekly") == "cs"
        assert classify_recording("Weekly sync") == "cs"
        assert classify_recording("WEEKLY review") == "cs"

    def test_cs_monthly(self):
        """Should classify titles containing 'monthly' as CS."""
        assert classify_recording("Monthly business review") == "cs"
        assert classify_recording("Acme Corp monthly") == "cs"
        assert classify_recording("MONTHLY sync") == "cs"

    def test_skip_internal(self):
        """Should classify titles containing 'internal' as skip."""
        assert classify_recording("Internal team sync") == "skip"
        assert classify_recording("internal sync") == "skip"
        assert classify_recording("INTERNAL planning") == "skip"

    def test_skip_standup(self):
        """Should classify titles containing standup/stand-up as skip."""
        assert classify_recording("Engineering standup") == "skip"
        assert classify_recording("Daily stand-up") == "skip"
        assert classify_recording("Team Stand Up") == "skip"
        assert classify_recording("STANDUP notes") == "skip"

    def test_cs_takes_priority_over_internal(self):
        """CS keywords should take priority over 'internal' or 'standup'."""
        assert classify_recording("Internal customer success review") == "cs"
        assert classify_recording("Internal weekly check-in") == "cs"

    def test_sales_default(self):
        """Non-CS, non-internal titles should be classified as sales."""
        assert classify_recording("Product roadmap review") == "sales"
        assert classify_recording("Sales call") == "sales"
        assert classify_recording("checking inventory") == "sales"

    def test_skip_empty_and_none(self):
        """Should classify empty/None titles as skip."""
        assert classify_recording("") == "skip"
        assert classify_recording(None) == "skip"


class TestShouldProcessRecording:
    """Tests for should_process_recording backward-compat wrapper."""

    def test_returns_true_for_cs(self):
        assert should_process_recording("Customer Success Call") is True

    def test_returns_true_for_sales(self):
        assert should_process_recording("Sales call") is True

    def test_returns_false_for_skip(self):
        assert should_process_recording("Internal meeting") is False
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
