"""
Auto-Resume: Kaggle完了を待ち、Claude Codeを1回だけ起動して終了する。

仕組み:
    1. kaggle_ops.py monitor を呼ぶ（ブロッキング: 完了まで待機）
    2. state.json の auto_resume_remaining を確認
       - > 0 → デクリメント → claude -p 起動 → 自然終了
       - = 0 → claude を起動せず終了（ループ停止）
       - 未設定 → 無制限（後方互換）
    3. スクリプト自然終了（何も残らない）

Claude終了後のstate確認 (パイプライン未完了検知):
    - waiting_training になった → Kaggle提出成功 → 終了
    - resume_from が BUILD_PHASES (architect等) → 途中終了 → 再起動
    - それ以外 (evaluatorが停止判断、またはNone) → 正常終了 → 再起動しない

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


def _release_if_owner():
    """
    ロックファイルが自プロセス(PID一致)のものである場合のみ解放する。

    release_lock() は常にファイルを削除するため、別プロセス(AR2等)が
    ロックを取得している状態で呼ぶと誤って削除してしまう。
    このヘルパーはPIDを確認してから解放することでその問題を防ぐ。
    """
    try:
        if not LOCK_FILE.exists():
            return
        lock_data = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
        if lock_data.get("pid") == os.getpid():
            LOCK_FILE.unlink(missing_ok=True)
        # else: 別プロセスのlock → 削除しない
    except Exception:
        pass  # 読み取り失敗時は安全側に倒してスキップ


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
# Constants
# ---------------------------------------------------------------------------

# Claude がここで止まった場合 = evaluator が判断する前にパイプラインが中断
# → auto_resume が再起動すべきフェーズ
BUILD_PHASES = {"architect", "researcher", "builder_data", "datachecker", "builder_model"}

# Claude 終了後のパイプライン未完了検知で最大何回再起動するか
MAX_PIPELINE_RELAUNCH = 2


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _build_evaluator_prompt(state: dict, kaggle_status: str, error_type: str = "", error_context: str = "") -> str:
    """
    evaluator 向けプロンプトを生成する。
    retry_context が存在する場合は「残り回数確認・改善余地判断」の指示を含める。
    evaluator が続行/停止を自律的に決定できるようにする。
    """
    feature = state.get("current_feature", "?")
    attempt = state.get("current_attempt", "?")
    resume_from = state.get("resume_from", "evaluator")

    if kaggle_status == "error":
        return (
            f"Kaggle training for {feature} attempt {attempt} FAILED "
            f"(error_type={error_type}). "
            f"error_context: {error_context[:300] if error_context else 'see state.json'}. "
            f"state.json has been updated with resume_from=builder_model. "
            f"Fix the notebook error in notebooks/{feature}_{attempt}/train.ipynb, "
            f"then call submit_and_monitor(max_loops=1) to re-submit and keep the auto-resume chain alive. "
            f"NEVER use submit() alone — it does not spawn auto_resume and breaks the automation loop."
        )

    # retry_context の情報を読み込んでプロンプトに含める
    retry_ctx = (
        state.get("real_rate_retry_context")
        or state.get("retry_context")
        or {}
    )
    max_attempt = retry_ctx.get("max_attempt")

    chain_note = ""
    if max_attempt is not None:
        attempts_left = max_attempt - int(attempt)
        if attempts_left <= 0:
            # 上限到達 → evaluator に停止を指示
            chain_note = (
                f" NOTE: current_attempt={attempt} has reached max_attempt={max_attempt}. "
                f"After evaluating, declare the best attempt as final and STOP the chain. "
                f"Do NOT submit another attempt."
            )
        else:
            # 残りあり → evaluatorが改善余地を判断
            automation_test = retry_ctx.get("automation_test", False)
            ctx_note = retry_ctx.get("note", "")
            if automation_test:
                # 自動化テストモード: 必ず次のattemptを続行
                chain_note = (
                    f" NOTE: current_attempt={attempt}, max_attempt={max_attempt} "
                    f"({attempts_left} attempt(s) remaining). "
                    f"AUTOMATION TEST MODE: ALWAYS continue to next attempt regardless of metrics. "
                    f"Do NOT declare no_further_improvement. "
                    f"continue to builder_model -> submit_and_monitor(max_loops=1). "
                    f"Context: {ctx_note}"
                )
            else:
                chain_note = (
                    f" NOTE: current_attempt={attempt}, max_attempt={max_attempt} "
                    f"({attempts_left} attempt(s) remaining). "
                    f"After evaluating: if improvement is still plausible, "
                    f"continue to architect -> builder_model -> submit_and_monitor(max_loops=1). "
                    f"If no further improvement is likely (e.g. consistent degradation), "
                    f"declare the best attempt as final and STOP. "
                    f"The evaluator decides - do not blindly continue to next attempt."
                )

    return (
        f"Kaggle training for {feature} attempt {attempt} has {kaggle_status}. "
        f"Results have been fetched and state.json updated. "
        f"Resume from {resume_from}.{chain_note}"
    )


def _build_relaunch_prompt(state: dict) -> str:
    """
    パイプライン途中（BUILD_PHASES）で止まった場合の再起動プロンプト。
    明示的に「submission まで続けろ」と指示する。
    """
    feature = state.get("current_feature", "?")
    attempt = state.get("current_attempt", "?")
    resume_from = state.get("resume_from", "architect")

    retry_ctx = (
        state.get("real_rate_retry_context")
        or state.get("retry_context")
        or {}
    )
    max_attempt = retry_ctx.get("max_attempt")
    max_note = f" (max_attempt={max_attempt})" if max_attempt else ""

    return (
        f"Pipeline incomplete: state.json shows resume_from={resume_from} for "
        f"{feature} attempt {attempt}{max_note}. "
        f"Continue the pipeline from {resume_from}: "
        f"{resume_from} -> builder_model -> submit_and_monitor(max_loops=1). "
        f"Do NOT exit until submit_and_monitor() has been called "
        f"(state.json status must reach 'waiting_training')."
    )


def _launch_with_pipeline_check(initial_prompt: str, max_relaunch: int = MAX_PIPELINE_RELAUNCH) -> None:
    """
    Claudeを起動し、終了後にパイプライン完了を確認する。
    BUILD_PHASESで止まった場合は最大 max_relaunch 回まで再起動する。
    evaluatorが停止判断した場合（resume_from not in BUILD_PHASES）は再起動しない。
    """
    prompt = initial_prompt
    for attempt_num in range(max_relaunch + 1):
        _release_if_owner()
        log.info("Lock released (owner-checked). Launching Claude...")
        launched = launch_claude(prompt)

        if not launched:
            log.error("Claude Code failed to launch. Manual resume needed.")
            return

        # Claude終了後にstateを再確認
        state_after = load_state()
        status_after = state_after.get("status")
        resume_after = state_after.get("resume_from")

        if status_after == "waiting_training":
            log.info("Claude reached waiting_training (submitted to Kaggle). Cycle complete.")
            return

        if resume_after in BUILD_PHASES and attempt_num < max_relaunch:
            # パイプライン途中で止まった → evaluatorはまだ判断していない → 再起動
            log.warning(
                f"Claude exited at resume_from='{resume_after}' without submitting "
                f"(relaunch {attempt_num + 1}/{max_relaunch}). Relaunching..."
            )
            prompt = _build_relaunch_prompt(state_after)
        else:
            # evaluatorが停止判断した、または再起動上限に達した
            if resume_after in BUILD_PHASES:
                log.error(
                    f"Claude exited at resume_from='{resume_after}' after {max_relaunch} relaunch(es). "
                    f"Manual resume needed."
                )
            else:
                # automation_test: evaluatorが早期停止したが attempts_left > 0 → 強制継続
                retry_ctx_after = (
                    state_after.get("real_rate_retry_context")
                    or state_after.get("retry_context")
                    or {}
                )
                auto_test = retry_ctx_after.get("automation_test", False)
                max_att = retry_ctx_after.get("max_attempt")
                cur_att = state_after.get("current_attempt", 0)
                left = (max_att - int(cur_att)) if (auto_test and max_att) else 0

                if auto_test and left > 0 and status_after == "completed" and attempt_num < max_relaunch:
                    log.warning(
                        f"automation_test: Evaluator declared completed early (attempts_left={left}). "
                        f"Force-continuing to next attempt (relaunch {attempt_num + 1}/{max_relaunch})."
                    )
                    # reset state to in_progress to allow submission
                    state_after["status"] = "in_progress"
                    state_after["resume_from"] = "builder_model"
                    save_state(state_after)
                    force_prompt = (
                        f"AUTOMATION TEST OVERRIDE: The evaluator declared the project complete at attempt {cur_att}, "
                        f"but max_attempt={max_att} requires continuing to attempt {cur_att + 1}. "
                        f"IGNORE the previous no_further_improvement decision. "
                        f"Build and submit meta_model attempt {cur_att + 1} now. "
                        f"Use submit_and_monitor(max_loops=1). "
                        f"Context from retry_context: {retry_ctx_after.get('improvement_mandate', '')}"
                    )
                    prompt = force_prompt
                    # continue the for-loop to relaunch
                    continue
                else:
                    log.info(
                        f"Claude completed (status={status_after}, resume_from={resume_after}). "
                        f"Evaluator made stop/continue decision."
                    )
            return


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def run_monitor(interval: int = 60, max_hours: float = 3.0, initial_wait: int = 90) -> bool:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "kaggle_ops.py"),
        "monitor",
        "--interval", str(interval),
        "--initial-wait", str(initial_wait),
    ]
    log.info(f"Starting monitor (interval={interval}s, max={max_hours}h, initial_wait={initial_wait}s)")
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
    # On Windows, .CMD files must be run via cmd /c to be interpreted as shell scripts
    if claude_cmd.upper().endswith(".CMD"):
        cmd_args = ["cmd", "/c", claude_cmd, "-p", prompt]
    else:
        cmd_args = [claude_cmd, "-p", prompt]
    # Remove CLAUDECODE env var: claude refuses to start inside another claude session
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    try:
        result = subprocess.run(
            cmd_args,
            cwd=str(PROJECT_ROOT),
            timeout=3600,
            env=env,
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

    if status == "in_progress" and (state.get("auto_resume_remaining") or 0) > 0:
        # Training already complete, results already fetched — skip Kaggle polling
        # and launch Claude directly (handles the case where claude previously failed)
        resume_from = state.get("resume_from", "evaluator")
        remaining = state.get("auto_resume_remaining")
        log.info(f"State=in_progress, resume_from={resume_from}, remaining={remaining}. Launching Claude directly.")
        if not acquire_lock():
            log.info("Another auto_resume is already running. Exiting.")
            return
        try:
            if remaining is not None:
                state["auto_resume_remaining"] = remaining - 1
                save_state(state)
                log.info(f"auto_resume_remaining: {remaining} → {remaining - 1}")
            prompt = _build_evaluator_prompt(state, kaggle_status="complete")
            # _launch_with_pipeline_check releases lock before launching Claude,
            # then relaunches if Claude stops mid-pipeline (BUILD_PHASES).
            _launch_with_pipeline_check(prompt)
        finally:
            _release_if_owner()  # PID確認してから解放 (AR2のlockを誤削除しない)
            log.info("auto_resume exiting.")
        return

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
        # initial_wait=120: 新カーネルはQUEUEDからRUNNINGに移行するまで90秒以上かかることがある。
        # 90秒だと古いエラーステータスを誤読みしてfalse-positive errorが発生する。
        ok = run_monitor(interval=args.interval, max_hours=args.max_hours, initial_wait=120)

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

        # automation_test mode: compute attempts_left from retry_context
        retry_ctx = (
            state.get("real_rate_retry_context")
            or state.get("retry_context")
            or {}
        )
        automation_test = retry_ctx.get("automation_test", False)
        max_attempt_ctx = retry_ctx.get("max_attempt")
        current_attempt_ctx = state.get("current_attempt", 0)
        attempts_left = (
            (max_attempt_ctx - int(current_attempt_ctx))
            if (automation_test and max_attempt_ctx is not None)
            else None
        )

        if remaining is not None and remaining <= 0:
            # automation_test: もし attempts_left > 0 なら残り回数を補充して継続
            if automation_test and attempts_left is not None and attempts_left > 0:
                log.info(
                    f"automation_test: auto_resume_remaining=0 but attempts_left={attempts_left}. "
                    f"Resetting remaining to 1 to continue chain."
                )
                remaining = 1
                state["auto_resume_remaining"] = remaining
                save_state(state)
            else:
                log.info(f"auto_resume_remaining={remaining}. NOT launching Claude. Loop stopped.")
                if ok:
                    log.info("Results are fetched. Run manually: claude -p 'Resume from evaluator'")
                else:
                    log.info(f"Error fix needed. Run manually: claude -p 'Resume from builder_model'")
                return

        # 7. Decrement counter
        # エラー時は残り回数を消費しない（エラー修正は attempt消費に含めない）
        # automation_test mode: attempts_left>0 の間は残り回数を消費しない（1に保つ）
        if remaining is not None:
            if kaggle_status == "error":
                # エラーは attempt カウント対象外 → デクリメントしない
                log.info(f"Kaggle ERROR: auto_resume_remaining NOT decremented (kept at {remaining})")
            elif automation_test and attempts_left is not None and attempts_left > 0:
                new_remaining = max(1, remaining - 1)  # 少なくとも1は残す
                log.info(f"automation_test: auto_resume_remaining kept at {new_remaining} (attempts_left={attempts_left})")
                state["auto_resume_remaining"] = new_remaining
                save_state(state)
            else:
                new_remaining = remaining - 1
                log.info(f"auto_resume_remaining: {remaining} → {new_remaining}")
                state["auto_resume_remaining"] = new_remaining
                save_state(state)

        # 8. Fire Claude Code (with pipeline completion check)
        prompt = _build_evaluator_prompt(
            state,
            kaggle_status=kaggle_status,
            error_type=error_type,
            error_context=error_context,
        )
        # _launch_with_pipeline_check releases lock before launching Claude,
        # then relaunches if Claude stops mid-pipeline (BUILD_PHASES).
        # Does NOT relaunch if evaluator decided to stop (resume_from not in BUILD_PHASES).
        _launch_with_pipeline_check(prompt)

    finally:
        _release_if_owner()  # PID確認してから解放 (AR2のlockを誤削除しない)
        log.info("auto_resume exiting.")


if __name__ == "__main__":
    main()
