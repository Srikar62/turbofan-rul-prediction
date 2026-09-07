"""
Dual data source loader for the dashboard.
1. Tries to load pre-computed results from dashboard_data/results.json
2. Falls back to live inference from .pt model files (lazy import to avoid
   heavy PyTorch / pipeline imports when JSON cache is available)
Provides a refresh function for the dashboard UI button.
"""
import json
import os
import sys

# Add project root to path so we can import pipeline modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RESULTS_PATH = os.path.join(PROJECT_ROOT, 'dashboard_data', 'results.json')


def load_results():
    """
    Load results from JSON cache if available, otherwise run live inference.
    Returns: dict with all metrics, predictions, expert gates, dataset info.
    """
    if os.path.exists(RESULTS_PATH):
        print(f'[Dashboard] Loading cached results from {RESULTS_PATH}')
        with open(RESULTS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)

    print('[Dashboard] No cached results found — running live inference...')
    return refresh_results()


def refresh_results():
    """
    Force re-run inference from .pt models and update the JSON cache.
    Uses lazy import to avoid loading the full ML pipeline unless needed.
    Returns: dict with all metrics, predictions, expert gates, dataset info.
    """
    # Lazy import to keep dashboard startup fast
    from export_results import _retrain_and_export
    return _retrain_and_export()


def get_results_age():
    """Return how old the cached results are, or None if no cache exists."""
    if not os.path.exists(RESULTS_PATH):
        return None
    import datetime
    with open(RESULTS_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    ts = data.get('metadata', {}).get('timestamp')
    if ts:
        created = datetime.datetime.fromisoformat(ts)
        return datetime.datetime.now() - created
    return None
