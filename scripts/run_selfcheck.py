from models.qwen_victim import load_qwen, write_startup_selfcheck


if __name__ == "__main__":
    ctx = load_qwen()
    write_startup_selfcheck(ctx, "results/qwen_startup_selfcheck.txt")
