import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

EXAMPLE_DIRS = (
    Path("examples"),
    Path("docs") / "examples",
)

def discover_examples(extra_dirs: Optional[List[Path]] = None) -> List[Path]:
    """
    Discover example files (.py and .ipynb) under default and extra directories.
    Returns a list of Paths.
    """
    dirs = list(EXAMPLE_DIRS)
    if extra_dirs:
        dirs.extend(extra_dirs)

    files: List[Path] = []
    for base in dirs:
        if base.is_dir():
            files.extend(sorted(base.rglob("*.py")))
            files.extend(sorted(base.rglob("*.ipynb")))
    return files

def run_example(path: Path, timeout: int = 10, env: Optional[Dict[str, str]] = None) -> Tuple[int, str, str]:
    """
    Execute a Python example in a subprocess.
    Returns (returncode, stdout, stderr).
    """
    cmd = [sys.executable, str(path)]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=str(path.parent),
    )
    return proc.returncode, proc.stdout, proc.stderr