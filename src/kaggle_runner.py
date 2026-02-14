"""
Kaggle API Runner - 学習Notebookの投入・状態確認・結果取得
オーケストレーターが直接使う共通ユーティリティ
"""

import subprocess
import json
import time
import shutil
import tempfile
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up Kaggle credentials for KGAT_ token format
if 'KAGGLE_API_TOKEN' in os.environ:
    os.environ['KAGGLE_KEY'] = os.environ['KAGGLE_API_TOKEN']


def _make_tmp_dir(prefix: str) -> Path:
    """OS非依存の一時ディレクトリを作成する"""
    return Path(tempfile.mkdtemp(prefix=prefix))


def push_notebook(feature: str, attempt: int) -> dict:
    """Kaggle Notebookを投入する"""
    notebook_dir = f"notebooks/{feature}_{attempt}"

    if not Path(notebook_dir).exists():
        raise FileNotFoundError(f"{notebook_dir} が存在しない")

    result = subprocess.run(
        ["kaggle", "kernels", "push", "-p", notebook_dir],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Kaggle push failed: {result.stderr}")

    # kernel-metadata.json からkernel IDを取得
    with open(f"{notebook_dir}/kernel-metadata.json") as f:
        meta = json.load(f)

    return {
        "kernel_id": meta["id"],
        "submitted_at": datetime.now().isoformat(),
        "status": "submitted"
    }


def check_status(kernel_id: str) -> str:
    """Kaggle Notebookの実行状態を確認する
    Returns: "running" / "complete" / "error" / "queued"
    """
    result = subprocess.run(
        ["kaggle", "kernels", "status", kernel_id],
        capture_output=True, text=True
    )

    output = result.stdout.strip().lower()

    if "complete" in output:
        return "complete"
    elif "error" in output or "fail" in output:
        return "error"
    elif "running" in output:
        return "running"
    elif "queued" in output:
        return "queued"
    else:
        return f"unknown: {output}"


def fetch_results(kernel_id: str, feature: str, attempt: int) -> dict:
    """Kaggle Notebookの出力を取得し所定パスに配置する"""

    # OS非依存の一時ディレクトリに出力をダウンロード
    tmp_dir = _make_tmp_dir(f"kaggle_output_{feature}_{attempt}_")

    result = subprocess.run(
        ["kaggle", "kernels", "output", kernel_id, "-p", str(tmp_dir)],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(f"Kaggle output fetch failed: {result.stderr}")

    # ファイルを所定パスに配置
    files_placed = {}

    # submodel_output.csv
    src = tmp_dir / "submodel_output.csv"
    if src.exists():
        dst = f"data/submodel_outputs/{feature}.csv"
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        files_placed["output"] = dst

    # model.pt
    src = tmp_dir / "model.pt"
    if src.exists():
        dst = f"models/submodels/{feature}/model.pt"
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        files_placed["model"] = dst

    # training_result.json
    src = tmp_dir / "training_result.json"
    if src.exists():
        dst = f"logs/training/{feature}_{attempt}.json"
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        files_placed["log"] = dst

    # 一時ディレクトリ削除
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "status": "fetched",
        "files": files_placed,
        "fetched_at": datetime.now().isoformat()
    }


def fetch_error_log(kernel_id: str) -> str:
    """エラー時のログを取得する"""
    tmp_dir = _make_tmp_dir("kaggle_error_")

    result = subprocess.run(
        ["kaggle", "kernels", "output", kernel_id, "-p", str(tmp_dir)],
        capture_output=True, text=True
    )

    # 標準出力/エラーを結合して返す
    logs = []
    if tmp_dir.exists():
        for f in tmp_dir.iterdir():
            if f.is_file():
                logs.append(f"=== {f.name} ===\n{f.read_text()}")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return "\n".join(logs) if logs else result.stdout + result.stderr


def wait_and_fetch(kernel_id: str, feature: str, attempt: int,
                   poll_interval: int = 30, max_wait: int = 3600) -> dict:
    """学習完了を待って結果を取得する（PCを開いたまま待つ場合用）"""

    elapsed = 0
    while elapsed < max_wait:
        status = check_status(kernel_id)

        if status == "complete":
            return fetch_results(kernel_id, feature, attempt)
        elif status == "error":
            error_log = fetch_error_log(kernel_id)
            return {"status": "error", "log": error_log}

        print(f"  Status: {status} ({elapsed}s elapsed)")
        time.sleep(poll_interval)
        elapsed += poll_interval

    return {"status": "timeout", "elapsed": elapsed}
