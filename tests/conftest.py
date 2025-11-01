from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
# Ensure both the repository root (for imports like `src.utils...`) and
# the actual source directory (for `utils...`) are importable.
root_str = str(repo_root)
src_str = str(repo_root / "src")
if root_str not in sys.path:
	sys.path.insert(0, root_str)
if src_str not in sys.path:
	sys.path.insert(0, src_str)