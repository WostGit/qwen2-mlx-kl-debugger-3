import subprocess


if __name__ == "__main__":
    subprocess.check_call(["python", "run_all.py", "--qwen-startup-selfcheck", "--debug-llm"])
    subprocess.check_call(["python", "scripts/summarize_results.py"])
    subprocess.check_call(["python", "scripts/plot_results.py"])
