"""
Auto-Resume: Kaggle完了を待ち、Claude Codeを1回だけ起動して終了する。

仕組み:
    1. kaggle_ops.py monitor を呼ぶ（ブロッキング: 完了まで待機）
    2. state.json の auto_resume_remaining を確認
       - > 0 → デクリメント → claude -p 起動 → 自然終了
       - = 0 → claude を起動せず終了（ループ停止）
       - 未設定 → 無制限（後方互換）
    3. スクリプト自然終了（何も残らない）

使い方:
    # submit後にバックグラウンド起動（submit_and_monitor が自動で起動）
    pythonw scripts/auto_resume.py &

    # 回数制限付き（3回だけ自動再開）
    pythonw scripts/auto_resume.py --max-loops 3 &

    # 状態確認
    python scripts/auto_resume.py --status

    # 強制停止（次のclaude起動を止める）
    python scripts/auto_resume.py --stop
"""

import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = PROJECT_ROOT / "shared" / "state.json"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "auto_resume.log"
LOCK_FILE = LOG_DIR / "auto_resume.lock"

# --- Logging ---
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("auto_resume")


# ---------------------------------------------------------------------------
# Lock file
# ---------------------------------------------------------------------------
def acquire_lock() -> bool:
    if LOCK_FILE.exists():
        try:
            age_sec = (datetime.now() - datetime.fromtimestamp(LOCK_FILE.stat().st_mtime)).total_seconds()
            if age_sec > 14400:
                log.warning(f"Stale lock ({age_sec/3600:.1f}h old). Removing.")
                LOCK_FILE.unlink()
            else:
                return False
        except Exception:
            return False
    try:
        LOCK_FILE.write_text(
            json.dumps({"pid": os.getpid(), "started": datetime.now().isoformat()}),
            encoding="utf-8",
        )
        return True
    except Exception as e:
        log.error(f"Lock creation failed: {e}")
        return False


def release_lock():
    try:
        LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------
def load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: dict):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def run_monitor(interval: int = 60, max_hours: float = 3.0) -> bool:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "kaggle_ops.py"),
        "monitor",
        "--interval", str(interval),
    ]
    log.info(f"Starting monitor (interval={interval}s, max={max_hours}h)")
    try:
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=max_hours * 3600)
        log.info(f"Monitor exited (code={result.returncode})")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        log.error(f"Monitor timed out after {max_hours}h")
        return False
    except Exception as e:
        log.error(f"Monitor failed: {e}")
        return False


def find_claude_executable() -> str | None:
    """Find the claude executable, checking PATH and known fallback locations."""
    import shutil
    claude_cmd = shutil.which("claude")
    if claude_cmd:
        return claude_cmd
    # Windows fallback: npm global install locations
    fallback_paths = [
        r"C:\Users\tatuk\AppData\Roaming\npm\claude.CMD",
        r"C:\Users\tatuk\AppData\Roaming\npm\claude.cmd",
    ]
    for fp in fallback_paths:
        if os.path.exists(fp):
            log.info(f"Found claude at fallback path: {fp}")
            return fp
    return None


def launch_claude(prompt: str) -> bool:
    claude_cmd = find_claude_executable()
    if claude_cmd is None:
        log.error("'claude' not found in PATH or fallback locations")
        return False
    log.info(f"Launching {claude_cmd} -p ...")
    try:
        result = subprocess.run(
            [claude_cmd, "-p", prompt],
            cwd=str(PROJECT_ROOT),
            timeout=3600,
        )
        log.info(f"Claude Code exited (code={result.returncode})")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        log.error("Claude Code timed out (1h)")
        return False
    except FileNotFoundError:
        log.error(f"'{claude_cmd}' not found or failed to execute")
        return False
    except Exception as e:
        log.error(f"Claude launch failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Wait for Kaggle, then fire Claude once")
    parser.add_argument("--status", action="store_true", help="Show current status and exit")
    parser.add_argument("--stop", action="store_true", help="Set remaining to 0 (stop auto-resume loop)")
    parser.add_argument("--max-loops", type=int, default=None,
                        help="Set max auto-resume loops (saved to state.json). "
                             "Omit for unlimited.")
    parser.add_argument("--interval", type=int, default=60, help="Poll interval seconds (default: 60)")
    parser.add_argument("--max-hours", type=float, default=3.0, help="Max wait hours (default: 3)")
    args = parser.parse_args()

    # --- --status ---
    if args.status:
        state = load_state()
        locked = LOCK_FILE.exists()
        lock_info = ""
        if locked:
            try:
                lock_data = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
                lock_info = f" (pid={lock_data.get('pid')}, since {lock_data.get('started', '?')})"
            except Exception:
                pass
        remaining = state.get("auto_resume_remaining")
        remaining_str = "unlimited" if remaining is None else str(remaining)
        print(f"state.json status : {state.get('status', 'unknown')}")
        print(f"feature/attempt   : {state.get('current_feature', '?')} #{state.get('current_attempt', '?')}")
        print(f"Kaggle kernel     : {state.get('kaggle_kernel', 'none')}")
        print(f"auto_resume       : {'RUNNING' + lock_info if locked else 'not running'}")
        print(f"remaining loops   : {remaining_str}")
        return

    # --- --stop ---
    if args.stop:
        state = load_state()
        state["auto_resume_remaining"] = 0
        save_state(state)
        print("auto_resume_remaining set to 0. Next cycle will not launch Claude.")
        return

    # --- Main flow ---
    log.info("=" * 60)
    log.info("auto_resume started")

    # 1. Validate state
    state = load_state()
    status = state.get("status")
    feature = state.get("current_feature", "?")
    attempt = state.get("current_attempt", "?")

    if status != "waiting_training":
        log.info(f"Not waiting_training (status={status}). Exiting.")
        return

    # 2. Set max-loops if specified (first launch only)
    if args.max_loops is not None:
        state["auto_resume_remaining"] = args.max_loops
        save_state(state)
        log.info(f"auto_resume_remaining set to {args.max_loops}")

    # 3. Lock
    if not acquire_lock():
        log.info("Another auto_resume is already running. Exiting.")
        return

    try:
        log.info(f"Watching: {feature} attempt {attempt}")
        log.info(f"Kernel: {state.get('kaggle_kernel')}")

        remaining = state.get("auto_resume_remaining")
        log.info(f"Remaining loops: {'unlimited' if remaining is None else remaining}")

        # 4. Block until Kaggle completes
        ok = run_monitor(interval=args.interval, max_hours=args.max_hours)

        # 5. Re-read state (monitor updated it)
        state = load_state()
        resume_from = state.get("resume_from", "evaluator")
        error_type = state.get("error_type", "unknown")
        error_context = state.get("error_context", "")

        if not ok:
            # Training failed — check if monitor properly classified the error
            if state.get("status") == "in_progress" and resume_from == "builder_model":
                log.warning(f"Training FAILED (error_type={error_type}). Launching Claude to fix.")
            else:
                log.error("Monitor exited unexpectedly. Manual resume needed.")
                return
            kaggle_status = "error"
        else:
            kaggle_status = "complete"

        # 6. Check remaining count
        remaining = state.get("auto_resume_remaining")

        if remaining is not None and remaining <= 0:
            log.info(f"auto_resume_remaining={remaining}. NOT launching Claude. Loop stopped.")
            if ok:
                log.info("Results are fetched. Run manually: claude -p 'Resume from evaluator'")
            else:
                log.info(f"Error fix needed. Run manually: claude -p 'Resume from builder_model'")
            return

        # 7. Decrement counter
        if remaining is not None:
            state["auto_resume_remaining"] = remaining - 1
            save_state(state)
            log.info(f"auto_resume_remaining: {remaining} → {remaining - 1}")

        # 8. Fire Claude Code exactly once
        if kaggle_status == "error":
            prompt = (
                f"Kaggle training for {feature} attempt {attempt} FAILED "
                f"(error_type={error_type}). "
                f"error_context: {error_context[:300] if error_context else 'see state.json'}. "
                f"state.json has been updated with resume_from=builder_model. "
                f"Fix the notebook error and re-submit."
            )
        else:
            prompt = (
                f"Kaggle training for {feature} attempt {attempt} has {kaggle_status}. "
                f"Results have been fetched and state.json updated. "
                f"Resume from {resume_from}."
            )
        launched = launch_claude(prompt)

        if launched:
            log.info("Done. Claude Code session completed.")
        else:
            log.error("Claude Code failed. Manual resume needed.")

    finally:
        release_lock()
        log.info("auto_resume exiting.")


if __name__ == "__main__":
    main()
