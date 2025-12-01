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


class Leaderboard:
    """
    Manages the UncensorBench leaderboard on HuggingFace Spaces.

    Example usage:
        ```python
        from uncensorbench.leaderboard import Leaderboard

        # View leaderboard (no token needed)
        lb = Leaderboard()
        df = lb.get_dataframe()
        print(df)

        # Submit results (token required)
        lb = Leaderboard(token="hf_...")
        lb.submit({
            "model": "my-model",
            "model_family": "Llama",
            "model_size": "8B",
            "method": "abliteration",  # none, abliteration, steering, finetuning, prompting, other
            "uncensored_rate": 0.15,
            "avg_compliance_score": 0.23,
            "total_prompts": 150,
        })
        ```

    View the live leaderboard at:
    https://huggingface.co/spaces/wisent-ai/UncensorBench-Leaderboard
    """

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
    
    def get_entries(self) -> List[Dict[str, Any]]:
        """
        Get all leaderboard entries.
        
        Returns:
            List of leaderboard entries sorted by uncensored_rate (descending)
        """
        entries = self._download_leaderboard()
        return sorted(entries, key=lambda x: x.get("uncensored_rate", 0), reverse=True)
    
    def get_dataframe(self):
        """
        Get leaderboard as a pandas DataFrame.

        Returns:
            pandas DataFrame with leaderboard data, or None if pandas not installed
        """
        try:
            import pandas as pd

            entries = self.get_entries()
            if not entries:
                return pd.DataFrame()

            # Flatten the data for DataFrame
            rows = []
            for entry in entries:
                row = {
                    "model": entry.get("model"),
                    "model_family": entry.get("model_family"),
                    "model_size": entry.get("model_size"),
                    "method": entry.get("method"),
                    "uncensored_rate": entry.get("uncensored_rate"),
                    "avg_compliance_score": entry.get("avg_compliance_score"),
                    "total_prompts": entry.get("total_prompts"),
                    "timestamp": entry.get("timestamp"),
                    "submitter": entry.get("submitter"),
                    "sample_responses_url": entry.get("sample_responses_url"),
                }
                rows.append(row)

            df = pd.DataFrame(rows)
            return df.sort_values("uncensored_rate", ascending=False).reset_index(drop=True)
        except ImportError:
            print("pandas is required for DataFrame output. Install with: pip install pandas")
            return None
    
    def submit(self, entry: Dict[str, Any], replace: bool = True):
        """
        Submit a new entry to the leaderboard.

        Args:
            entry: Leaderboard entry with model results
            replace: If True, replace existing entry for same model (default: True)

        Required fields in entry:
            - model: Model name/identifier (e.g., "meta-llama/Llama-3.1-8B-Instruct")
            - model_family: Model family (e.g., "Llama", "Qwen", "Mistral")
            - model_size: Model size (e.g., "8B", "70B", "1.5B")
            - method: Censorship removal method (any string accepted, e.g., "none", "abliteration",
                      "steering", "finetuning", "prompting", "my_custom_method")
            - uncensored_rate: Float between 0 and 1
            - avg_compliance_score: Float between 0 and 1
            - total_prompts: Number of prompts evaluated

        Optional fields:
            - submitter: Name/handle of the submitter
            - timestamp: ISO format timestamp (auto-generated if not provided)
            - sample_responses_url: URL to JSON file with sample model responses for analysis
        """
        if not self.token:
            raise ValueError(
                "HuggingFace token is required for submissions. "
                "Initialize Leaderboard with token='hf_...'"
            )

        # Validate required fields
        required_fields = [
            "model", "model_family", "model_size", "method",
            "uncensored_rate", "avg_compliance_score", "total_prompts"
        ]
        for field in required_fields:
            if field not in entry:
                raise ValueError(f"Missing required field: {field}")

        # Warn about unknown methods (but allow them)
        if entry["method"] not in KNOWN_METHODS:
            print(f"Note: '{entry['method']}' is not a known method. "
                  f"Known methods: {', '.join(KNOWN_METHODS)}")

        # Ensure timestamp
        if "timestamp" not in entry:
            entry["timestamp"] = datetime.now().isoformat()

        # Get current leaderboard
        entries = self._download_leaderboard()

        # Handle replacement
        if replace:
            entries = [e for e in entries if e.get("model") != entry["model"]]

        # Add new entry
        entries.append(entry)

        # Sort by uncensored_rate
        entries = sorted(entries, key=lambda x: x.get("uncensored_rate", 0), reverse=True)

        # Upload
        self._upload_leaderboard(entries)

        print(f"Submitted {entry['model']} to leaderboard")
        print(f"  Model Family: {entry['model_family']}")
        print(f"  Model Size: {entry['model_size']}")
        print(f"  Method: {entry['method']}")
        print(f"  Uncensored Rate: {entry['uncensored_rate']:.1%}")
        print(f"  Avg Compliance: {entry['avg_compliance_score']:.3f}")
        if entry.get('sample_responses_url'):
            print(f"  Sample Responses: {entry['sample_responses_url']}")
        print(f"View at: https://huggingface.co/spaces/{self.space_id}")
    
    def remove(self, model: str):
        """
        Remove an entry from the leaderboard.
        
        Args:
            model: Model name to remove
        """
        if not self.token:
            raise ValueError("HuggingFace token is required for removals.")
        
        entries = self._download_leaderboard()
        original_count = len(entries)
        entries = [e for e in entries if e.get("model") != model]
        
        if len(entries) == original_count:
            print(f"Model '{model}' not found in leaderboard")
            return
        
        self._upload_leaderboard(entries)
        print(f"Removed '{model}' from leaderboard")
    
    def print_leaderboard(self, top_n: int = 20):
        """Print a formatted leaderboard to console."""
        entries = self.get_entries()[:top_n]
        
        if not entries:
            print("Leaderboard is empty")
            return
        
        print("=" * 80)
        print("UncensorBench Leaderboard")
        print("=" * 80)
        print(f"{'Rank':<6} {'Model':<40} {'Uncensored':<12} {'Compliance':<12}")
        print("-" * 80)
        
        for i, entry in enumerate(entries, 1):
            model = entry.get("model", "Unknown")[:38]
            rate = entry.get("uncensored_rate", 0)
            compliance = entry.get("avg_compliance_score", 0)
            print(f"{i:<6} {model:<40} {rate:>10.1%} {compliance:>10.3f}")
        
        print("=" * 80)
