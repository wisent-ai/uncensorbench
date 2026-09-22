"""Optional public leaderboard client."""

from .operations import LeaderboardOperations

class Leaderboard(LeaderboardOperations):
    """Read and submit UncensorBench leaderboard entries."""

__all__ = ["Leaderboard"]
