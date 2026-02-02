# Services module for Lido CS Automation

from .tldv_client import (
    TldvClient,
    TldvAPIError,
    TldvAuthenticationError,
    TldvAuthorizationError,
    TldvNotFoundError,
    TldvValidationError,
    TldvRateLimitError,
    Recording,
    TranscriptSegment,
    Speaker,
    should_process_recording,
)

from .transcript_analyzer import (
    TranscriptAnalyzer,
    TranscriptAnalyzerError,
)

from .sheets_client import (
    SheetsClient,
    SheetsClientError,
    SheetsAuthenticationError,
    SheetsAPIError,
)

from .knowledge_base_writer import (
    append_qa_pairs,
    get_qa_count,
    KnowledgeBaseError,
)

__all__ = [
    # tldv client
    "TldvClient",
    "TldvAPIError",
    "TldvAuthenticationError",
    "TldvAuthorizationError",
    "TldvNotFoundError",
    "TldvValidationError",
    "TldvRateLimitError",
    "Recording",
    "TranscriptSegment",
    "Speaker",
    "should_process_recording",
    # transcript analyzer
    "TranscriptAnalyzer",
    "TranscriptAnalyzerError",
    # sheets client
    "SheetsClient",
    "SheetsClientError",
    "SheetsAuthenticationError",
    "SheetsAPIError",
    # knowledge base
    "append_qa_pairs",
    "get_qa_count",
    "KnowledgeBaseError",
]
