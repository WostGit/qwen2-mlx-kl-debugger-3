from models.qwen_victim import sanity_check_llm
from pathlib import Path

sanity_check_llm(Path("results/qwen_sanity_check.jsonl"))
