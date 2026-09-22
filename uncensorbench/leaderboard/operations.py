"""Local leaderboard operations."""

from datetime import datetime
from typing import Any, Dict, List

from .client import KNOWN_METHODS, LeaderboardClient

class LeaderboardOperations(LeaderboardClient):
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
