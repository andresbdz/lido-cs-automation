"""
Knowledge base writer for appending Q&A pairs to markdown file.

Thread-safe file operations for concurrent access.
"""

import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default knowledge base path (relative to project root)
DEFAULT_KB_PATH = "knowledge_base.md"

# Thread lock for file operations
_file_lock = threading.Lock()


class KnowledgeBaseError(Exception):
    """Base exception for knowledge base errors."""
    pass


def append_qa_pairs(
    qa_pairs: list[dict],
    call_title: str,
    call_date: str,
    kb_path: Optional[str] = None,
) -> int:
    """
    Append Q&A pairs to the knowledge base markdown file.

    Thread-safe: Uses a lock to prevent concurrent write conflicts.

    Args:
        qa_pairs: List of Q&A dictionaries with 'question', 'answer', 'topic' keys.
        call_title: Title of the call for the section header.
        call_date: Date of the call (YYYY-MM-DD format).
        kb_path: Path to knowledge base file. Defaults to knowledge_base.md in project root.

    Returns:
        Number of Q&A pairs appended.

    Raises:
        KnowledgeBaseError: If file operations fail.
    """
    if not qa_pairs:
        logger.info("No Q&A pairs to append")
        return 0

    # Determine file path
    if kb_path is None:
        # Find project root (where knowledge_base.md should be)
        kb_path = _find_knowledge_base_path()

    logger.info(f"Appending {len(qa_pairs)} Q&A pairs to {kb_path}")

    # Format the content
    content = _format_qa_section(qa_pairs, call_title, call_date)

    # Thread-safe file append
    with _file_lock:
        try:
            _append_to_file(kb_path, content)
            logger.info(f"Successfully appended {len(qa_pairs)} Q&A pairs")
            return len(qa_pairs)

        except OSError as e:
            logger.error(f"Failed to write to knowledge base: {e}")
            raise KnowledgeBaseError(f"Failed to write to file: {e}")


def _find_knowledge_base_path() -> str:
    """
    Find the knowledge base file path.

    Searches for knowledge_base.md in common locations.

    Returns:
        Absolute path to knowledge base file.
    """
    # Check common locations
    search_paths = [
        Path.cwd() / DEFAULT_KB_PATH,
        Path(__file__).parent.parent / DEFAULT_KB_PATH,
    ]

    for path in search_paths:
        if path.exists():
            return str(path.resolve())

    # Default to current directory
    return str(Path.cwd() / DEFAULT_KB_PATH)


def _format_qa_section(
    qa_pairs: list[dict],
    call_title: str,
    call_date: str,
) -> str:
    """
    Format Q&A pairs into markdown section.

    Args:
        qa_pairs: List of Q&A dictionaries.
        call_title: Title for the section header.
        call_date: Date for the section header.

    Returns:
        Formatted markdown string.
    """
    # Format date for display
    try:
        date_obj = datetime.fromisoformat(call_date.split("T")[0])
        formatted_date = date_obj.strftime("%Y-%m-%d")
    except (ValueError, AttributeError):
        formatted_date = call_date

    # Build section
    lines = [
        "",
        f"## [{formatted_date}] - {call_title}",
        "",
    ]

    for qa in qa_pairs:
        question = qa.get("question", "").strip()
        answer = qa.get("answer", "").strip()
        topic = qa.get("topic", "General").strip()

        lines.extend([
            f"**Q:** {question}",
            "",
            f"**A:** {answer}",
            "",
            f"**Topic:** {topic}",
            "",
            "---",
            "",
        ])

    return "\n".join(lines)


def _append_to_file(file_path: str, content: str) -> None:
    """
    Append content to file, creating if it doesn't exist.

    Args:
        file_path: Path to the file.
        content: Content to append.
    """
    # Ensure directory exists
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    # Check if file exists and has content
    file_exists = os.path.exists(file_path)
    needs_newline = False

    if file_exists:
        with open(file_path, "r", encoding="utf-8") as f:
            existing_content = f.read()
            # Add newline if file doesn't end with one
            if existing_content and not existing_content.endswith("\n"):
                needs_newline = True

    # Append to file
    with open(file_path, "a", encoding="utf-8") as f:
        if needs_newline:
            f.write("\n")
        f.write(content)


def get_qa_count(kb_path: Optional[str] = None) -> int:
    """
    Count the number of Q&A pairs in the knowledge base.

    Args:
        kb_path: Path to knowledge base file.

    Returns:
        Number of Q&A pairs (counted by "**Q:**" occurrences).
    """
    if kb_path is None:
        kb_path = _find_knowledge_base_path()

    if not os.path.exists(kb_path):
        return 0

    with open(kb_path, "r", encoding="utf-8") as f:
        content = f.read()

    return content.count("**Q:**")


def clear_knowledge_base(kb_path: Optional[str] = None) -> None:
    """
    Reset knowledge base to initial state (header only).

    Use with caution - this deletes all Q&A content.

    Args:
        kb_path: Path to knowledge base file.
    """
    if kb_path is None:
        kb_path = _find_knowledge_base_path()

    with _file_lock:
        with open(kb_path, "w", encoding="utf-8") as f:
            f.write("# Lido Customer Success Knowledge Base\n")

    logger.info(f"Knowledge base cleared: {kb_path}")
