"""
Transcript analyzer using Anthropic's Claude API.

Extracts structured insights from customer success call transcripts.
"""

import json
import logging
import os
import re
import time
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)

# Model configuration
MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 4096


class TranscriptAnalyzerError(Exception):
    """Base exception for transcript analyzer errors."""
    pass


class TranscriptAnalyzer:
    """
    Analyzes customer success call transcripts using Claude.

    Usage:
        analyzer = TranscriptAnalyzer()
        next_steps = analyzer.extract_next_steps(transcript, "2024-01-15")
        qa_pairs = analyzer.extract_qa_pairs(transcript)
    """

    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3):
        """
        Initialize the transcript analyzer.

        Args:
            api_key: Anthropic API key. If not provided, reads from ANTHROPIC_API_KEY env var.
            max_retries: Maximum number of retries for API failures.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise TranscriptAnalyzerError(
                "API key not provided. Set ANTHROPIC_API_KEY environment variable."
            )

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.max_retries = max_retries

    def _call_api_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> str:
        """
        Call Claude API with retry logic for transient failures.

        Args:
            system_prompt: System message for the model.
            user_prompt: User message containing the task.
            temperature: Sampling temperature (0.0 for deterministic).

        Returns:
            Model response text.

        Raises:
            TranscriptAnalyzerError: If all retries fail.
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"API call attempt {attempt + 1}/{self.max_retries}")

                response = self.client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )

                return response.content[0].text

            except anthropic.RateLimitError as e:
                last_error = e
                wait_time = min(60, 2 ** (attempt + 1))
                logger.warning(f"Rate limited. Waiting {wait_time}s before retry.")
                time.sleep(wait_time)

            except anthropic.APIStatusError as e:
                last_error = e
                if e.status_code >= 500:
                    # Server error - retry
                    wait_time = 2 ** attempt
                    logger.warning(f"Server error {e.status_code}. Retrying in {wait_time}s.")
                    time.sleep(wait_time)
                else:
                    # Client error - don't retry
                    logger.error(f"API error: {e}")
                    raise TranscriptAnalyzerError(f"API error: {e}")

            except anthropic.APIConnectionError as e:
                last_error = e
                wait_time = 2 ** attempt
                logger.warning(f"Connection error. Retrying in {wait_time}s.")
                time.sleep(wait_time)

        raise TranscriptAnalyzerError(f"API call failed after {self.max_retries} retries: {last_error}")

    def _parse_json_response(self, response: str) -> dict | list:
        """
        Parse JSON from model response, handling markdown code blocks.

        Args:
            response: Raw model response text.

        Returns:
            Parsed JSON data.
        """
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response.strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}\nResponse: {response[:500]}")
            raise TranscriptAnalyzerError(f"Failed to parse model response as JSON: {e}")

    def extract_next_steps(self, transcript: str, call_date: str) -> dict:
        """
        Extract actionable next steps from a call transcript.

        The next steps are structured using MECE principles:
        - Mutually Exclusive: Each action item is distinct, no overlap
        - Collectively Exhaustive: All discussed action items are captured

        Args:
            transcript: Full call transcript text.
            call_date: Date of the call (ISO format, e.g., "2024-01-15").

        Returns:
            Dictionary with:
                - prospect_email: Email address of the prospect (or None if not found)
                - call_date: The provided call date
                - next_steps: MECE-structured action items as formatted string
                - due_date: Suggested due date for follow-up
        """
        logger.info(f"Extracting next steps from transcript (call_date: {call_date})")

        system_prompt = """You are an expert customer success analyst. Your task is to extract structured, actionable next steps from call transcripts.

Follow MECE principles for next steps:
- MUTUALLY EXCLUSIVE: Each action item must be distinct with no overlap
- COLLECTIVELY EXHAUSTIVE: Capture ALL discussed commitments and action items

Output valid JSON only, no additional text."""

        user_prompt = f"""Analyze this customer success call transcript and extract next steps.

CALL DATE: {call_date}

TRANSCRIPT:
{transcript}

Extract and return a JSON object with these fields:
{{
    "prospect_email": "<email address mentioned in transcript, or null if not found>",
    "call_date": "{call_date}",
    "next_steps": "<MECE-structured action items as a formatted list>",
    "due_date": "<suggested follow-up date based on discussed timelines, ISO format>"
}}

For next_steps, format as a numbered list with:
1. WHO is responsible (us/prospect)
2. WHAT specific action
3. WHEN it should be done (if mentioned)

Example next_steps format:
"1. [Lido Team] Send API documentation for spreadsheet integration - by EOD Friday
2. [Prospect] Share current workflow spreadsheet for review - by next Monday
3. [Lido Team] Schedule follow-up demo with technical team - within 2 weeks"

If no email is explicitly mentioned in the transcript, set prospect_email to null.
For due_date, suggest a reasonable follow-up date (typically 3-7 business days after call) if not explicitly discussed.

Return ONLY the JSON object."""

        response = self._call_api_with_retry(system_prompt, user_prompt)
        result = self._parse_json_response(response)

        # Ensure required fields exist
        return {
            "prospect_email": result.get("prospect_email"),
            "call_date": result.get("call_date", call_date),
            "next_steps": result.get("next_steps", "No next steps identified"),
            "due_date": result.get("due_date"),
        }

    def extract_qa_pairs(self, transcript: str) -> list[dict]:
        """
        Extract Q&A pairs about Lido product usage from the transcript.

        Only extracts questions about "how to do XYZ in Lido" with detailed answers.
        These are used to build the customer success knowledge base.

        Args:
            transcript: Full call transcript text.

        Returns:
            List of dictionaries, each containing:
                - question: The question asked (normalized form)
                - answer: Detailed answer provided
                - topic: Category/topic for the Q&A
        """
        logger.info("Extracting Q&A pairs from transcript")

        system_prompt = """You are an expert at identifying valuable product knowledge from customer calls.

Your task is to extract Q&A pairs that would be useful for a knowledge base.

Focus ONLY on:
- "How do I..." questions about using Lido
- "Can I..." questions about Lido features
- "What's the best way to..." questions about Lido workflows
- Technical questions about Lido integrations, formulas, or automations

DO NOT include:
- Pricing/business questions
- General small talk
- Questions not about Lido product usage
- Questions with vague or incomplete answers

Output valid JSON only."""

        user_prompt = f"""Analyze this call transcript and extract Q&A pairs about Lido product usage.

TRANSCRIPT:
{transcript}

Extract questions where someone asked "how to do something in Lido" and received a substantive answer.

Return a JSON array of objects:
[
    {{
        "question": "<normalized question about Lido usage>",
        "answer": "<detailed answer that was provided>",
        "topic": "<category: one of 'Formulas', 'Integrations', 'Automations', 'Data Sources', 'Sharing', 'Templates', 'General'>"
    }}
]

Guidelines:
- Normalize questions to be clear and reusable (e.g., "How do I connect Google Sheets to Lido?")
- Include full context in the answer - make it useful for someone reading it later
- Only include Q&As where a real answer was provided (not "I'll get back to you")
- If no relevant Q&A pairs exist, return an empty array []

Return ONLY the JSON array."""

        response = self._call_api_with_retry(system_prompt, user_prompt)
        result = self._parse_json_response(response)

        # Validate result is a list
        if not isinstance(result, list):
            logger.warning(f"Expected list, got {type(result)}. Returning empty list.")
            return []

        # Validate each item has required fields
        validated = []
        for item in result:
            if all(key in item for key in ("question", "answer", "topic")):
                validated.append({
                    "question": item["question"],
                    "answer": item["answer"],
                    "topic": item["topic"],
                })
            else:
                logger.warning(f"Skipping invalid Q&A item: {item}")

        logger.info(f"Extracted {len(validated)} Q&A pairs")
        return validated

    def summarize_call(self, transcript: str) -> str:
        """
        Generate a concise summary of the call.

        Args:
            transcript: Full call transcript text.

        Returns:
            Brief summary of the call (2-3 paragraphs).
        """
        logger.info("Generating call summary")

        system_prompt = """You are an expert at summarizing customer success calls.
Write clear, concise summaries that capture the key points."""

        user_prompt = f"""Summarize this customer success call in 2-3 short paragraphs.

Include:
- Main topics discussed
- Key decisions or outcomes
- Any concerns or blockers raised
- Overall sentiment (positive/neutral/concerned)

TRANSCRIPT:
{transcript}

Provide the summary directly, no JSON formatting needed."""

        return self._call_api_with_retry(system_prompt, user_prompt, temperature=0.3)
