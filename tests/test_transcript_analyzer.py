"""Tests for transcript analyzer."""

import pytest
from unittest.mock import Mock, patch
from services.transcript_analyzer import TranscriptAnalyzer, TranscriptAnalyzerError


class TestTranscriptAnalyzerInit:
    """Tests for TranscriptAnalyzer initialization."""

    def test_requires_api_key(self, monkeypatch):
        """Should raise error when no API key provided."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(TranscriptAnalyzerError):
            TranscriptAnalyzer()

    def test_accepts_api_key_param(self):
        """Should accept API key as parameter."""
        with patch("services.transcript_analyzer.anthropic.Anthropic"):
            analyzer = TranscriptAnalyzer(api_key="test-key")
            assert analyzer.api_key == "test-key"


class TestParseJsonResponse:
    """Tests for JSON parsing helper."""

    @pytest.fixture
    def analyzer(self):
        with patch("services.transcript_analyzer.anthropic.Anthropic"):
            return TranscriptAnalyzer(api_key="test-key")

    def test_parses_plain_json(self, analyzer):
        """Should parse plain JSON."""
        result = analyzer._parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parses_json_in_code_block(self, analyzer):
        """Should extract JSON from markdown code blocks."""
        response = '```json\n{"key": "value"}\n```'
        result = analyzer._parse_json_response(response)
        assert result == {"key": "value"}

    def test_parses_json_in_plain_code_block(self, analyzer):
        """Should extract JSON from plain code blocks."""
        response = '```\n{"key": "value"}\n```'
        result = analyzer._parse_json_response(response)
        assert result == {"key": "value"}

    def test_raises_on_invalid_json(self, analyzer):
        """Should raise error on invalid JSON."""
        with pytest.raises(TranscriptAnalyzerError):
            analyzer._parse_json_response("not valid json")


class TestExtractNextSteps:
    """Tests for extract_next_steps method."""

    @pytest.fixture
    def mock_analyzer(self):
        with patch("services.transcript_analyzer.anthropic.Anthropic"):
            analyzer = TranscriptAnalyzer(api_key="test-key")
            return analyzer

    def test_returns_expected_structure(self, mock_analyzer):
        """Should return dict with required fields."""
        mock_response = '''```json
{
    "prospect_email": "test@example.com",
    "call_date": "2024-01-15",
    "next_steps": "1. [Lido] Send docs\\n2. [Prospect] Review",
    "due_date": "2024-01-22"
}
```'''
        mock_analyzer._call_api_with_retry = Mock(return_value=mock_response)

        result = mock_analyzer.extract_next_steps("test transcript", "2024-01-15")

        assert "prospect_email" in result
        assert "call_date" in result
        assert "next_steps" in result
        assert "due_date" in result
        assert result["prospect_email"] == "test@example.com"

    def test_handles_missing_email(self, mock_analyzer):
        """Should return None for prospect_email when not found."""
        mock_response = '''```json
{
    "prospect_email": null,
    "call_date": "2024-01-15",
    "next_steps": "1. Follow up",
    "due_date": "2024-01-22"
}
```'''
        mock_analyzer._call_api_with_retry = Mock(return_value=mock_response)

        result = mock_analyzer.extract_next_steps("test transcript", "2024-01-15")

        assert result["prospect_email"] is None


class TestExtractQaPairs:
    """Tests for extract_qa_pairs method."""

    @pytest.fixture
    def mock_analyzer(self):
        with patch("services.transcript_analyzer.anthropic.Anthropic"):
            analyzer = TranscriptAnalyzer(api_key="test-key")
            return analyzer

    def test_returns_list_of_qa_pairs(self, mock_analyzer):
        """Should return list of Q&A dictionaries."""
        mock_response = '''```json
[
    {
        "question": "How do I connect Google Sheets?",
        "answer": "Go to Integrations > Google Sheets > Connect",
        "topic": "Integrations"
    }
]
```'''
        mock_analyzer._call_api_with_retry = Mock(return_value=mock_response)

        result = mock_analyzer.extract_qa_pairs("test transcript")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["question"] == "How do I connect Google Sheets?"
        assert result[0]["topic"] == "Integrations"

    def test_returns_empty_list_when_no_qa(self, mock_analyzer):
        """Should return empty list when no Q&A found."""
        mock_response = "[]"
        mock_analyzer._call_api_with_retry = Mock(return_value=mock_response)

        result = mock_analyzer.extract_qa_pairs("test transcript")

        assert result == []

    def test_filters_invalid_items(self, mock_analyzer):
        """Should filter out items missing required fields."""
        mock_response = '''[
            {"question": "Valid?", "answer": "Yes", "topic": "General"},
            {"question": "Missing answer"}
        ]'''
        mock_analyzer._call_api_with_retry = Mock(return_value=mock_response)

        result = mock_analyzer.extract_qa_pairs("test transcript")

        assert len(result) == 1
        assert result[0]["question"] == "Valid?"
