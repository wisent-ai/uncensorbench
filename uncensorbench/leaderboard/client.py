"""
Leaderboard management for UncensorBench.

Manages submissions and retrieval of benchmark results on HuggingFace Spaces.
"""

import csv
import io
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path


LEADERBOARD_SPACE = "wisent-ai/UncensorBench-Leaderboard"
LEADERBOARD_FILE = "leaderboard.csv"

# CSV columns in order
CSV_COLUMNS = [
    "model", "model_family", "model_size", "method",
    "uncensored_rate", "avg_compliance_score", "total_prompts",
    "timestamp", "submitter", "sample_responses_url"
]

# Known censorship removal methods (any method string is accepted)
KNOWN_METHODS = [
    "none",           # Baseline (no modification)
    "abliteration",   # Abliteration technique
    "steering",       # Steering vectors
    "finetuning",     # Fine-tuning based
    "prompting",      # Prompt-based jailbreaking
    "other",          # Other methods
]


class LeaderboardClient:
    """Remote CSV transport for the optional leaderboard."""

    def __init__(self, token: Optional[str] = None, space_id: str = LEADERBOARD_SPACE):
        """
        Initialize the leaderboard.

        Args:
            token: HuggingFace API token (required for submissions)
            space_id: HuggingFace Space ID
        """
        self.token = token
        self.space_id = space_id
        self._api = None

    @property
    def api(self):
        """Lazy load HuggingFace API."""
        if self._api is None:
            try:
                from huggingface_hub import HfApi
                self._api = HfApi(token=self.token)
            except ImportError:
                raise ImportError(
                    "huggingface_hub is required for leaderboard features. "
                    "Install with: pip install huggingface_hub"
                )
        return self._api

    def _download_leaderboard(self) -> List[Dict[str, Any]]:
        """Download the current leaderboard data from the Space."""
        try:
            from huggingface_hub import hf_hub_download

            local_path = hf_hub_download(
                repo_id=self.space_id,
                filename=LEADERBOARD_FILE,
                repo_type="space",
                token=self.token,
            )

            entries = []
            with open(local_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert numeric fields
                    if row.get("uncensored_rate"):
                        row["uncensored_rate"] = float(row["uncensored_rate"])
                    if row.get("avg_compliance_score"):
                        row["avg_compliance_score"] = float(row["avg_compliance_score"])
                    if row.get("total_prompts"):
                        row["total_prompts"] = int(row["total_prompts"])
                    entries.append(row)
            return entries
        except Exception:
            # Leaderboard doesn't exist yet or is empty
            return []

    def _upload_leaderboard(self, entries: List[Dict[str, Any]]):
        """Upload the leaderboard data to the HuggingFace Space."""
        import tempfile
        import os

        # Sort by uncensored_rate descending
        entries = sorted(entries, key=lambda x: x.get("uncensored_rate", 0), reverse=True)

        # Create CSV content
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(entries)
        csv_content = output.getvalue()

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = f.name

        try:
            self.api.upload_file(
                path_or_fileobj=temp_path,
                path_in_repo=LEADERBOARD_FILE,
                repo_id=self.space_id,
                repo_type="space",
                commit_message=f"Update leaderboard: {datetime.now().isoformat()}",
            )
        finally:
            os.unlink(temp_path)
