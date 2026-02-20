"""Tests for duplicate recording prevention (race condition fix)."""

import json
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

# Import the module to patch
import app as app_module


class TestTryClaimRecording:
    """Tests for the atomic claim mechanism."""

    @pytest.fixture(autouse=True)
    def setup_temp_tracking_file(self, tmp_path):
        """Create a temporary tracking file for each test."""
        self.temp_file = tmp_path / "processed_recordings.json"
        self.lock = threading.Lock()

        # Patch at module level
        with patch.object(app_module, "PROCESSED_RECORDINGS_FILE", self.temp_file), \
             patch.object(app_module, "_tracking_lock", self.lock):
            yield

    def test_first_claim_succeeds(self):
        """First claim for a recording should succeed."""
        assert app_module.try_claim_recording("recording-123") is True

    def test_second_claim_fails(self):
        """Second claim for same recording should fail."""
        assert app_module.try_claim_recording("recording-123") is True
        assert app_module.try_claim_recording("recording-123") is False

    def test_different_recordings_both_succeed(self):
        """Claims for different recordings should both succeed."""
        assert app_module.try_claim_recording("recording-123") is True
        assert app_module.try_claim_recording("recording-456") is True

    def test_claim_sets_processing_status(self):
        """Claiming should set status to 'processing' with timestamp."""
        app_module.try_claim_recording("recording-123")

        data = app_module.load_processed_recordings()
        entry = data["recordings"]["recording-123"]

        assert entry["status"] == "processing"
        assert "claimed_at" in entry

    def test_completed_recording_cannot_be_claimed(self):
        """A completed recording cannot be claimed again."""
        app_module.try_claim_recording("recording-123")
        app_module.mark_as_processed("recording-123", "Test Title", {"sheets_updated": True})

        assert app_module.try_claim_recording("recording-123") is False

    def test_release_claim_allows_reclaim(self):
        """Releasing a claim should allow it to be claimed again."""
        assert app_module.try_claim_recording("recording-123") is True
        app_module.release_claim("recording-123")
        assert app_module.try_claim_recording("recording-123") is True

    def test_release_only_affects_processing_status(self):
        """Release should only work on 'processing' status, not 'completed'."""
        app_module.try_claim_recording("recording-123")
        app_module.mark_as_processed("recording-123", "Test Title", {})
        app_module.release_claim("recording-123")  # Should have no effect

        # Should still be completed
        assert app_module.is_already_processed("recording-123") is True

    def test_stale_claim_can_be_reclaimed(self):
        """A stale 'processing' entry should be reclaimable."""
        # Manually create a stale entry
        stale_time = datetime.now() - timedelta(
            seconds=app_module.STALE_PROCESSING_THRESHOLD_SECONDS + 60
        )
        data = {
            "recordings": {
                "recording-123": {
                    "status": "processing",
                    "claimed_at": stale_time.isoformat(),
                }
            }
        }
        with open(self.temp_file, "w") as f:
            json.dump(data, f)

        # Should be able to reclaim
        assert app_module.try_claim_recording("recording-123") is True

    def test_fresh_claim_cannot_be_reclaimed(self):
        """A fresh 'processing' entry should NOT be reclaimable."""
        app_module.try_claim_recording("recording-123")
        assert app_module.try_claim_recording("recording-123") is False

    def test_legacy_entries_treated_as_completed(self):
        """Entries without 'status' field should be treated as completed."""
        # Legacy format (before this fix)
        data = {
            "recordings": {
                "recording-123": {
                    "title": "Old Recording",
                    "processed_at": "2024-01-01T00:00:00",
                }
            }
        }
        with open(self.temp_file, "w") as f:
            json.dump(data, f)

        # Should not be claimable (treated as completed)
        assert app_module.try_claim_recording("recording-123") is False
        assert app_module.is_already_processed("recording-123") is True


class TestConcurrentClaims:
    """Tests for concurrent access (race condition simulation)."""

    @pytest.fixture(autouse=True)
    def setup_temp_tracking_file(self, tmp_path):
        """Create a temporary tracking file for each test."""
        self.temp_file = tmp_path / "processed_recordings.json"
        self.lock = threading.Lock()

        with patch.object(app_module, "PROCESSED_RECORDINGS_FILE", self.temp_file), \
             patch.object(app_module, "_tracking_lock", self.lock):
            yield

    def test_concurrent_claims_only_one_succeeds(self):
        """When multiple threads try to claim the same recording, only one should succeed."""
        recording_id = "concurrent-test-123"
        results = []
        results_lock = threading.Lock()
        num_threads = 10
        barrier = threading.Barrier(num_threads)

        def attempt_claim():
            barrier.wait()  # Synchronize all threads to start at the same time
            result = app_module.try_claim_recording(recording_id)
            with results_lock:
                results.append(result)

        threads = [threading.Thread(target=attempt_claim) for _ in range(num_threads)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly one thread should have succeeded
        success_count = sum(1 for r in results if r is True)
        assert success_count == 1, f"Expected 1 success, got {success_count}"

    def test_concurrent_claims_different_recordings_all_succeed(self):
        """Concurrent claims for different recordings should all succeed."""
        results = {}
        results_lock = threading.Lock()
        num_recordings = 10
        barrier = threading.Barrier(num_recordings)

        def attempt_claim(rec_id):
            barrier.wait()
            result = app_module.try_claim_recording(rec_id)
            with results_lock:
                results[rec_id] = result

        threads = [
            threading.Thread(target=attempt_claim, args=(f"recording-{i}",))
            for i in range(num_recordings)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should succeed
        assert all(results.values()), f"Expected all successes, got {results}"

    def test_high_contention_stress_test(self):
        """Stress test with many concurrent attempts on same recording."""
        recording_id = "stress-test-123"
        num_attempts = 50
        results = []
        results_lock = threading.Lock()
        barrier = threading.Barrier(num_attempts)

        def attempt_claim():
            barrier.wait()
            result = app_module.try_claim_recording(recording_id)
            with results_lock:
                results.append(result)

        with ThreadPoolExecutor(max_workers=num_attempts) as executor:
            futures = [executor.submit(attempt_claim) for _ in range(num_attempts)]
            for f in as_completed(futures):
                f.result()  # Raise any exceptions

        success_count = sum(1 for r in results if r is True)
        assert success_count == 1, f"Expected exactly 1 success in {num_attempts} attempts, got {success_count}"


class TestIsAlreadyProcessed:
    """Tests for is_already_processed function."""

    @pytest.fixture(autouse=True)
    def setup_temp_tracking_file(self, tmp_path):
        """Create a temporary tracking file for each test."""
        self.temp_file = tmp_path / "processed_recordings.json"
        self.lock = threading.Lock()

        with patch.object(app_module, "PROCESSED_RECORDINGS_FILE", self.temp_file), \
             patch.object(app_module, "_tracking_lock", self.lock):
            yield

    def test_unclaimed_recording_not_processed(self):
        """An unclaimed recording should not be considered processed."""
        assert app_module.is_already_processed("new-recording") is False

    def test_processing_recording_not_yet_processed(self):
        """A recording in 'processing' status should not be considered processed."""
        app_module.try_claim_recording("recording-123")
        assert app_module.is_already_processed("recording-123") is False

    def test_completed_recording_is_processed(self):
        """A recording in 'completed' status should be considered processed."""
        app_module.try_claim_recording("recording-123")
        app_module.mark_as_processed("recording-123", "Test", {})
        assert app_module.is_already_processed("recording-123") is True
