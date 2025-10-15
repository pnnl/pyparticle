from pathlib import Path
import sys

# Ensure repo root is on sys.path so 'tests.integration' imports always work
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
