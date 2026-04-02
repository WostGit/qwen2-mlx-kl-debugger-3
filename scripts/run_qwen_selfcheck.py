from models.qwen_victim import qwen_startup_selfcheck
from pathlib import Path

qwen_startup_selfcheck(Path("results/qwen_startup_selfcheck.txt"))
