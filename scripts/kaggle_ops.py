"""
Kaggle統一操作モジュール (Python API v2.0.0)

全Kaggle操作をPython APIに統一。Windows cp932エンコーディング問題を回避。

基本操作 (Python):
    from scripts.kaggle_ops import notebook_push, kernel_status, kernel_output
    from scripts.kaggle_ops import dataset_create, dataset_update

    result = notebook_push("notebooks/real_rate_1/")
    result = kernel_status("bigbigzabuton/gold-real-rate-1")
    result = kernel_output("bigbigzabuton/gold-real-rate-1", "data/outputs/")
    result = dataset_create("data/submodel_outputs/")
    result = dataset_update("data/dataset_upload_clean/", "v2: added file")

ワークフロー (Python):
    from scripts.kaggle_ops import submit_and_monitor

    # 提出 → state更新 → git commit → バックグラウンド監視
    result = submit_and_monitor(
        folder="notebooks/real_rate_1/",
        feature="real_rate",
        attempt=1,
    )

基本操作 (CLI):
    python scripts/kaggle_ops.py notebook-push notebooks/real_rate_1/
    python scripts/kaggle_ops.py kernel-status bigbigzabuton/gold-real-rate-1
    python scripts/kaggle_ops.py kernel-output bigbigzabuton/gold-real-rate-1 data/outputs/
    python scripts/kaggle_ops.py dataset-create data/submodel_outputs/
    python scripts/kaggle_ops.py dataset-update data/dataset_upload_clean/ "v2: notes"

ワークフロー (CLI):
    python scripts/kaggle_ops.py submit notebooks/real_rate_1/ real_rate 1
    python scripts/kaggle_ops.py monitor
    python scripts/kaggle_ops.py monitor --once
"""

import builtins
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# --- Windows cp932 encoding fix ---
# Kaggle API v2.0.0 uses open(file, ...) without encoding parameter.
# On Windows this defaults to cp932, which crashes on Unicode characters (e.g. ⚠).
# Patch builtins.open BEFORE importing kaggle to force UTF-8 for all text mode I/O.
_original_open = builtins.open


def _utf8_open(file, mode="r", *args, **kwargs):
    if isinstance(mode, str) and "b" not in mode and "encoding" not in kwargs:
        kwargs["encoding"] = "utf-8"
    return _original_open(file, mode, *args, **kwargs)


if sys.platform == "win32":
    builtins.open = _utf8_open

# --- Auth setup ---
from dotenv import load_dotenv

load_dotenv()

# Kaggle Python API expects KAGGLE_USERNAME + KAGGLE_KEY
# Our .env has KAGGLE_USERNAME + KAGGLE_API_TOKEN
if os.getenv("KAGGLE_API_TOKEN") and not os.getenv("KAGGLE_KEY"):
    os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_API_TOKEN")

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_enums import KernelWorkerStatus


@dataclass
class KaggleResult:
    """Kaggle操作の統一戻り値"""

    success: bool
    message: str
    data: dict = field(default_factory=dict)


def _get_api() -> KaggleApi:
    """認証済みKaggle APIインスタンスを取得"""
    api = KaggleApi()
    api.authenticate()
    return api


# ---------------------------------------------------------------------------
# 1. Notebook Push (新規・更新共用)
# ---------------------------------------------------------------------------
def notebook_push(folder: str) -> KaggleResult:
    """
    Kaggleにノートブックを提出する（新規・更新を自動判定）。

    Args:
        folder: kernel-metadata.json を含むディレクトリパス

    Returns:
        KaggleResult: success=True なら data["kernel_ref"] に提出先ID
    """
    folder = str(Path(folder).resolve())

    # バリデーション
    metadata_file = Path(folder) / "kernel-metadata.json"
    if not metadata_file.exists():
        return KaggleResult(False, f"kernel-metadata.json not found in {folder}")

    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    kernel_id = metadata.get("id")
    if not kernel_id:
        return KaggleResult(False, "kernel-metadata.json missing 'id' field")

    # train.ipynb or train.py の存在チェック
    code_file = metadata.get("code_file", "train.ipynb")
    if not (Path(folder) / code_file).exists():
        return KaggleResult(False, f"Code file '{code_file}' not found in {folder}")

    try:
        api = _get_api()
        response = api.kernels_push(folder)

        ref = getattr(response, "ref", None) or kernel_id
        url = getattr(response, "url", None) or f"https://www.kaggle.com/code/{ref}"

        # Verify the ref is actually reachable via the API.
        # When creating a NEW kernel, Kaggle may ignore the metadata `id` and
        # generate a slug from the title instead (e.g. "Gold Real Rate Model -
        # Attempt 8" → "gold-real-rate-model-attempt-8"), causing a mismatch.
        # If kernel_status returns 403, search kernels_list for the real slug.
        verified_ref = _resolve_actual_kernel_ref(ref, metadata.get("title", ""))
        if verified_ref != ref:
            print(f"  [INFO] Kaggle slug mismatch: metadata='{ref}' → actual='{verified_ref}'")
            ref = verified_ref
            url = f"https://www.kaggle.com/code/{ref}"

        return KaggleResult(
            True,
            f"Notebook pushed: {ref}",
            {"kernel_ref": ref, "url": url, "kernel_id": ref},
        )
    except Exception as e:
        error_msg = str(e)
        if "409" in error_msg or "Conflict" in error_msg:
            return KaggleResult(
                False,
                f"Kernel '{kernel_id}' already exists with a running/queued version. "
                "Wait for it to complete or delete it on Kaggle Web UI first.",
                {"kernel_id": kernel_id, "error_type": "conflict"},
            )
        return KaggleResult(False, f"Push failed: {e}")


# ---------------------------------------------------------------------------
# 1b. Kernel ref resolver (slug mismatch fix)
# ---------------------------------------------------------------------------
def _resolve_actual_kernel_ref(pushed_ref: str, metadata_title: str) -> str:
    """
    push後にAPIで実際のカーネルスラッグを検証する。

    Kaggleは新規カーネル作成時、metadata JSONの`id`を無視して
    タイトルからスラッグを生成することがある。この場合、
    `kernel_status(pushed_ref)` が 403 を返すので `kernels_list` で
    マッチするカーネルを探して正しいスラッグを返す。

    Args:
        pushed_ref: notebook_push が返した ref (metadata id ベース)
        metadata_title: kernel-metadata.json の "title" フィールド

    Returns:
        実際に使えるカーネルスラッグ (文字列)
    """
    # まず pushed_ref が正常かチェック
    sr = kernel_status(pushed_ref)
    if sr.success:
        return pushed_ref  # 問題なし

    # 403 等のエラー → kernels_list から正しい ref を探す
    try:
        api = _get_api()
        kernels = api.kernels_list(mine=True, page_size=20)
        title_lower = metadata_title.lower().strip()
        for k in kernels:
            k_title = getattr(k, "title", "") or ""
            k_ref = getattr(k, "ref", "") or ""
            if k_title.lower().strip() == title_lower and k_ref:
                return k_ref
    except Exception:
        pass

    # 見つからなければ元の ref を返す（後続でエラーになる）
    return pushed_ref


# ---------------------------------------------------------------------------
# 2. Kernel Status
# ---------------------------------------------------------------------------
_STATUS_MAP = {
    KernelWorkerStatus.QUEUED: "queued",
    KernelWorkerStatus.RUNNING: "running",
    KernelWorkerStatus.COMPLETE: "complete",
    KernelWorkerStatus.ERROR: "error",
    KernelWorkerStatus.CANCEL_REQUESTED: "cancel_requested",
    KernelWorkerStatus.CANCEL_ACKNOWLEDGED: "cancelled",
    KernelWorkerStatus.NEW_SCRIPT: "new_script",
}


def kernel_status(kernel_id: str) -> KaggleResult:
    """
    Kaggleカーネルの実行ステータスを確認する。

    Args:
        kernel_id: "username/kernel-slug" 形式

    Returns:
        KaggleResult: data["status"] に文字列ステータス、
                      data["failure_message"] にエラーメッセージ（あれば）
    """
    try:
        api = _get_api()
        response = api.kernels_status(kernel_id)

        status_enum = response.status
        status_str = _STATUS_MAP.get(status_enum, str(status_enum))
        failure_msg = response.failure_message or None

        return KaggleResult(
            True,
            f"Kernel {kernel_id}: {status_str}",
            {"status": status_str, "failure_message": failure_msg},
        )
    except Exception as e:
        return KaggleResult(False, f"Status check failed: {e}")


# ---------------------------------------------------------------------------
# 3. Kernel Output (結果取得)
# ---------------------------------------------------------------------------
def kernel_output(kernel_id: str, output_dir: str) -> KaggleResult:
    """
    完了したカーネルの出力ファイルをダウンロードする。

    Args:
        kernel_id: "username/kernel-slug" 形式
        output_dir: ダウンロード先ディレクトリ

    Returns:
        KaggleResult: data["files"] にダウンロードしたファイルパスのリスト
    """
    output_dir = str(Path(output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)

    try:
        api = _get_api()
        files, token = api.kernels_output(kernel_id, output_dir, quiet=True)

        return KaggleResult(
            True,
            f"Downloaded {len(files)} files to {output_dir}",
            {"files": files, "output_dir": output_dir},
        )
    except Exception as e:
        return KaggleResult(False, f"Output download failed: {e}")


# ---------------------------------------------------------------------------
# 4. Dataset Create (新規作成)
# ---------------------------------------------------------------------------
def dataset_create(folder: str) -> KaggleResult:
    """
    新規Kaggleデータセットを作成する。

    Args:
        folder: dataset-metadata.json を含むディレクトリパス

    Returns:
        KaggleResult: data["url"] にデータセットURL
    """
    folder = str(Path(folder).resolve())

    metadata_file = Path(folder) / "dataset-metadata.json"
    if not metadata_file.exists():
        return KaggleResult(False, f"dataset-metadata.json not found in {folder}")

    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    dataset_id = metadata.get("id")
    if not dataset_id:
        return KaggleResult(False, "dataset-metadata.json missing 'id' field")

    try:
        api = _get_api()
        response = api.dataset_create_new(folder, public=False, quiet=False, dir_mode="zip")

        url = getattr(response, "url", None) or f"https://www.kaggle.com/datasets/{dataset_id}"
        status = getattr(response, "status", None)
        status_str = str(status).lower() if status else ""

        if "error" in status_str:
            return KaggleResult(
                False,
                f"Dataset '{dataset_id}' already exists. Use dataset-update instead.",
                {"dataset_id": dataset_id, "hint": "use dataset_update", "status": status_str},
            )

        return KaggleResult(
            True,
            f"Dataset created: {dataset_id}",
            {"dataset_id": dataset_id, "url": url, "status": status_str},
        )
    except Exception as e:
        error_msg = str(e)
        if "already exists" in error_msg.lower() or "403" in error_msg:
            return KaggleResult(
                False,
                f"Dataset '{dataset_id}' already exists. Use dataset-update instead.",
                {"dataset_id": dataset_id, "hint": "use dataset_update"},
            )
        return KaggleResult(False, f"Dataset creation failed: {e}")


# ---------------------------------------------------------------------------
# 5. Dataset Update (バージョン更新)
# ---------------------------------------------------------------------------
def dataset_update(folder: str, version_notes: str) -> KaggleResult:
    """
    既存Kaggleデータセットの新バージョンを作成する。

    Args:
        folder: dataset-metadata.json を含むディレクトリパス
        version_notes: バージョンノート（例: "v3: added temporal_context.csv"）

    Returns:
        KaggleResult: data["url"] にデータセットURL
    """
    folder = str(Path(folder).resolve())

    metadata_file = Path(folder) / "dataset-metadata.json"
    if not metadata_file.exists():
        return KaggleResult(False, f"dataset-metadata.json not found in {folder}")

    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    dataset_id = metadata.get("id")
    if not dataset_id:
        return KaggleResult(False, "dataset-metadata.json missing 'id' field")

    try:
        api = _get_api()
        response = api.dataset_create_version(
            folder, version_notes=version_notes, quiet=False, dir_mode="zip"
        )

        url = getattr(response, "url", None) or f"https://www.kaggle.com/datasets/{dataset_id}"
        status = getattr(response, "status", None)

        return KaggleResult(
            True,
            f"Dataset updated: {dataset_id} ({version_notes})",
            {"dataset_id": dataset_id, "url": url, "status": str(status)},
        )
    except Exception as e:
        return KaggleResult(False, f"Dataset update failed: {e}")


# ===========================================================================
# Workflow: Submit and Monitor
# ===========================================================================

_PROJECT_ROOT = Path(__file__).parent.parent
_STATE_FILE = _PROJECT_ROOT / "shared" / "state.json"


def _load_state() -> dict:
    with open(_STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_state(state: dict):
    state["last_updated"] = datetime.now().isoformat()
    with open(_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def _git_commit_and_push(message: str) -> bool:
    try:
        subprocess.run(["git", "add", "-A"], cwd=_PROJECT_ROOT, check=True)
        result = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 and "nothing to commit" not in result.stdout:
            print(f"[WARN] git commit: {result.stdout.strip()}")
        subprocess.run(["git", "push"], cwd=_PROJECT_ROOT, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[WARN] git failed: {e}")
        return False


# ---------------------------------------------------------------------------
# 6. Submit (push + state update + git commit)
# ---------------------------------------------------------------------------
def submit(
    folder: str,
    feature: str,
    attempt: int,
) -> KaggleResult:
    """
    ノートブック提出 → state.json更新 → git commit & push。

    Args:
        folder: kernel-metadata.json を含むディレクトリパス
        feature: 特徴量名 (e.g. "real_rate", "meta_model")
        attempt: 試行番号
    """
    # Push
    push_result = notebook_push(folder)
    if not push_result.success:
        return push_result

    # Use kernel_ref (the verified actual Kaggle slug) not kernel_id (metadata value).
    # These can differ when Kaggle generates a slug from the title for new kernels.
    kernel_id = push_result.data.get("kernel_ref") or push_result.data.get("kernel_id", "")

    # Update state
    state = _load_state()
    state.update(
        {
            "status": "waiting_training",
            "resume_from": "evaluator",
            "kaggle_kernel": kernel_id,
            "submitted_at": datetime.now().isoformat(),
            "current_feature": feature,
            "current_attempt": attempt,
        }
    )
    _save_state(state)

    # Git commit & push
    _git_commit_and_push(f"kaggle: {feature} attempt {attempt} - submitted")

    url = push_result.data.get("url", "")
    print(f"\nKernel URL: {url}")
    print("Training is running on Kaggle cloud.")
    print('When ready, run: python scripts/kaggle_ops.py monitor')

    return KaggleResult(
        True,
        f"Submitted {feature} attempt {attempt}: {kernel_id}",
        {**push_result.data, "feature": feature, "attempt": attempt},
    )


# ---------------------------------------------------------------------------
# 7. Submit and Monitor (submit + background monitor)
# ---------------------------------------------------------------------------
def submit_and_monitor(
    folder: str,
    feature: str,
    attempt: int,
    max_loops: int | None = None,
) -> KaggleResult:
    """
    提出 → バックグラウンドで auto_resume.py を起動。

    auto_resume.py が monitor (ブロッキング) → 完了検知 → claude -p を1回起動 → 自然終了。
    claude -p 内で再度 submit_and_monitor() が呼ばれると新しい auto_resume.py が生まれ、
    max_loops 回まで自動ループする。

    Args:
        max_loops: 自動再開の最大回数。None=無制限、3=3回だけ自動実行。
    """
    result = submit(folder, feature, attempt)
    if not result.success:
        return result

    # Start background auto_resume (monitor + claude -p trigger)
    auto_resume_script = str(Path(__file__).resolve().parent / "auto_resume.py")
    cmd_args = [auto_resume_script]
    if max_loops is not None:
        cmd_args += ["--max-loops", str(max_loops)]

    try:
        if sys.platform == "win32":
            pythonw = str(Path(sys.executable).parent / "pythonw.exe")
            if not Path(pythonw).exists():
                pythonw = sys.executable
            subprocess.Popen(
                [pythonw] + cmd_args,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
                cwd=str(_PROJECT_ROOT),
            )
        else:
            subprocess.Popen(
                [sys.executable] + cmd_args,
                start_new_session=True,
                cwd=str(_PROJECT_ROOT),
            )
        loops_str = f"max {max_loops} loops" if max_loops is not None else "unlimited"
        print(f"[OK] auto_resume started ({loops_str})")
        print("     Status: python scripts/auto_resume.py --status")
    except Exception as e:
        print(f"[WARN] Could not start auto_resume: {e}")
        print(f"  Run manually: python scripts/auto_resume.py")

    return result


# ---------------------------------------------------------------------------
# 8. Monitor (poll until complete/error)
# ---------------------------------------------------------------------------

# Error classification for auto-retry decisions
_ERROR_PATTERNS = {
    "network_timeout": ["connection", "timeout", "refused"],
    "oom": ["memory", "killed", "oom"],
    "pandas_compat": ["fillna", "append", "asfreq"],
    "missing_dep": ["cannot import", "no module"],
    "dataset_missing": ["no such file", "target.csv", "gold-prediction"],
    "yfinance": ["multiindex", "per-column arrays"],
}


def _classify_error(log: str) -> str:
    log_lower = log.lower()
    for error_type, patterns in _ERROR_PATTERNS.items():
        if any(p in log_lower for p in patterns):
            return error_type
    return "unknown"


def monitor(
    check_interval: int = 60,
    max_hours: float = 3.0,
    once: bool = False,
    initial_wait: int = 90,
) -> KaggleResult:
    """
    state.json の kaggle_kernel を監視し、完了時に結果を取得する。

    Args:
        check_interval: チェック間隔（秒）
        max_hours: 最大監視時間
        once: True なら1回だけチェックして戻る

    完了時:
        - 結果を data/submodel_outputs/{feature}/ にダウンロード
        - state.json を "in_progress" + resume_from="evaluator" に更新
        - git commit & push

    エラー時:
        - エラー種別を分類
        - network_timeout → state に retry 情報を設定
        - それ以外 → state に error 情報を設定
        - git commit & push
    """
    state = _load_state()

    if state.get("status") != "waiting_training":
        return KaggleResult(
            False,
            f"Not in waiting_training state (current: {state.get('status')})",
        )

    kid = state.get("kaggle_kernel")
    feature = state.get("current_feature")
    attempt = state.get("current_attempt")

    if not kid:
        return KaggleResult(False, "No kaggle_kernel in state.json")

    print(f"Monitoring: {kid}")
    print(f"Feature: {feature}, Attempt: {attempt}")
    print(f"Interval: {check_interval}s, Max: {max_hours}h")
    if once:
        print("Mode: single check")
    print()

    # Wait before first poll to allow Kaggle to transition from previous
    # error/complete state to queued/running for the newly submitted version.
    # Without this delay, the monitor may detect the OLD failed version's status
    # and immediately exit with error, before the new version is even queued.
    if initial_wait > 0 and not once:
        print(f"Waiting {initial_wait}s for Kaggle to queue new submission...")
        time.sleep(initial_wait)

    start = datetime.now()
    max_wait = timedelta(hours=max_hours)
    check_count = 0

    while True:
        check_count += 1
        elapsed = datetime.now() - start
        print(f"[{datetime.now():%H:%M:%S}] Check #{check_count} ({elapsed.total_seconds()/60:.0f}m)")

        sr = kernel_status(kid)
        if not sr.success:
            print(f"  [WARN] {sr.message}")
            if once:
                return sr
            time.sleep(check_interval)
            continue

        s = sr.data["status"]

        if s == "complete":
            print(f"  [OK] Training COMPLETE!")

            # Download results
            output_dir = str(_PROJECT_ROOT / "data" / "submodel_outputs" / feature)
            dl = kernel_output(kid, output_dir)

            if dl.success:
                # Update state
                state = _load_state()
                state.update(
                    {
                        "status": "in_progress",
                        "resume_from": "evaluator",
                        "kaggle_kernel": kid,
                    }
                )
                _save_state(state)
                _git_commit_and_push(f"kaggle: {feature} attempt {attempt} - results fetched")

                print(f"  Results in: {output_dir}")
                print(f"  State updated: resume_from=evaluator")
                print(f'\n  Next: "Resume from where we left off"')

                return KaggleResult(
                    True,
                    f"Training complete. Results fetched for {feature} attempt {attempt}.",
                    {"status": "complete", "files": dl.data.get("files", [])},
                )
            else:
                return KaggleResult(False, f"Download failed: {dl.message}")

        elif s == "error":
            failure_msg = sr.data.get("failure_message") or ""
            error_type = _classify_error(failure_msg)
            print(f"  [FAIL] Training ERROR: {error_type}")
            print(f"  {failure_msg[:200]}")

            # Wait one more interval and re-check before treating as final error.
            # Kaggle may return the old version's error status while the new
            # version is still being queued. A second confirmation avoids
            # false positives that cause unnecessary claude relaunch cycles.
            if not once and check_count <= 3:
                print(f"  [WARN] Re-checking in {check_interval}s to confirm (attempt {check_count}/3)...")
                time.sleep(check_interval)
                sr2 = kernel_status(kid)
                if sr2.success and sr2.data["status"] != "error":
                    s2 = sr2.data["status"]
                    print(f"  [OK] Status changed to '{s2}' - was stale error. Continuing...")
                    continue  # Back to top of loop with updated status

            state = _load_state()
            state.update(
                {
                    "status": "in_progress",
                    "resume_from": "builder_model",
                    "error_context": failure_msg[:1000],
                    "error_type": error_type,
                }
            )
            _save_state(state)
            _git_commit_and_push(f"kaggle: {feature} attempt {attempt} - error ({error_type})")

            return KaggleResult(
                False,
                f"Training failed ({error_type})",
                {"status": "error", "error_type": error_type, "failure_message": failure_msg},
            )

        elif s in ("running", "queued", "new_script"):
            print(f"  ... {s}")

        else:
            print(f"  [WARN] Unknown status: {s}")

        if once:
            return KaggleResult(True, f"Status: {s}", {"status": s})

        if elapsed > max_wait:
            print(f"\n  [TIMEOUT] {max_hours}h exceeded")
            state = _load_state()
            state["user_action_required"] = True
            _save_state(state)
            return KaggleResult(False, f"Timeout after {max_hours}h. Kernel still {s}.")

        time.sleep(check_interval)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Kaggle unified operations tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Basic operations ---

    # notebook-push
    p = subparsers.add_parser("notebook-push", help="Submit notebook to Kaggle")
    p.add_argument("folder", help="Directory containing kernel-metadata.json")

    # kernel-status
    p = subparsers.add_parser("kernel-status", help="Check kernel execution status")
    p.add_argument("kernel_id", help="Kernel ID (username/slug)")

    # kernel-output
    p = subparsers.add_parser("kernel-output", help="Download kernel output files")
    p.add_argument("kernel_id", help="Kernel ID (username/slug)")
    p.add_argument("output_dir", help="Download destination directory")

    # dataset-create
    p = subparsers.add_parser("dataset-create", help="Create new Kaggle dataset")
    p.add_argument("folder", help="Directory containing dataset-metadata.json")

    # dataset-update
    p = subparsers.add_parser("dataset-update", help="Update existing dataset (new version)")
    p.add_argument("folder", help="Directory containing dataset-metadata.json")
    p.add_argument("version_notes", help="Version notes")

    # --- Workflow operations ---

    # submit (push + state + git)
    p = subparsers.add_parser("submit", help="Submit notebook + update state + git push")
    p.add_argument("folder", help="Directory containing kernel-metadata.json")
    p.add_argument("feature", help="Feature name (e.g. real_rate)")
    p.add_argument("attempt", type=int, help="Attempt number")
    p.add_argument("--monitor", action="store_true", help="Start background monitor after submit")
    p.add_argument("--max-loops", type=int, default=None,
                   help="Max auto-resume loops (requires --monitor). None=unlimited, 3=3 loops")

    # monitor
    p = subparsers.add_parser("monitor", help="Monitor running kernel until complete")
    p.add_argument("--once", action="store_true", help="Check once and exit")
    p.add_argument("--interval", type=int, default=60, help="Check interval in seconds (default: 60)")
    p.add_argument("--max-hours", type=float, default=3.0, help="Max monitoring hours (default: 3)")
    p.add_argument("--initial-wait", type=int, default=90,
                   help="Seconds to wait before first poll (default: 90). "
                        "Allows Kaggle to queue new submission before detecting old error state.")

    args = parser.parse_args()

    dispatch = {
        "notebook-push": lambda: notebook_push(args.folder),
        "kernel-status": lambda: kernel_status(args.kernel_id),
        "kernel-output": lambda: kernel_output(args.kernel_id, args.output_dir),
        "dataset-create": lambda: dataset_create(args.folder),
        "dataset-update": lambda: dataset_update(args.folder, args.version_notes),
        "submit": lambda: (
            submit_and_monitor(args.folder, args.feature, args.attempt, max_loops=args.max_loops)
            if args.monitor
            else submit(args.folder, args.feature, args.attempt)
        ),
        "monitor": lambda: monitor(
            check_interval=args.interval,
            max_hours=args.max_hours,
            once=args.once,
            initial_wait=args.initial_wait,
        ),
    }

    result = dispatch[args.command]()

    # Output
    status_icon = "OK" if result.success else "FAIL"
    print(f"\n[{status_icon}] {result.message}")
    if result.data:
        print(json.dumps(result.data, indent=2, ensure_ascii=False, default=str))

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
