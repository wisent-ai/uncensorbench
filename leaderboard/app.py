"""Dash entry point for the UncensorBench leaderboard."""

try:
    import dash
except ImportError as error:
    raise SystemExit("Leaderboard dashboard requires dash and dash-ag-grid. Install its dashboard dependencies before running this app.") from error

from dashboard.layout import build_layout
from dashboard import callbacks

app = dash.Dash(__name__, title="UncensorBench Leaderboard")
server = app.server
app.layout = build_layout()

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=7860)
