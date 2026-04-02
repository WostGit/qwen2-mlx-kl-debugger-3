from models.qwen_victim import load_qwen, run_sanity_check


if __name__ == "__main__":
    ctx = load_qwen()
    sanity = run_sanity_check(ctx)
    print(sanity)
    if not sanity["ok"]:
        raise SystemExit(1)
