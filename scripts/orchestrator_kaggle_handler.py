"""
Orchestrator用のKaggle統合ハンドラー

Kaggle提出 → 自動監視スクリプト起動 → セッション終了の一連の処理を提供
"""

import subprocess
import sys
import json
import os
import re
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


class KaggleSubmissionHandler:
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.state_file = self.project_root / 'shared' / 'state.json'

    def load_state(self):
        """state.jsonを読み込み"""
        with open(self.state_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def update_state(self, updates):
        """state.jsonを更新"""
        state = self.load_state()
        state.update(updates)
        state['last_updated'] = datetime.now().isoformat()

        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def extract_kernel_id(self, kaggle_output):
        """
        Kaggle push出力からkernel IDを抽出

        例: "Successfully pushed to username/gold-real-rate-1"
        → "username/gold-real-rate-1"
        """
        # パターン1: "Successfully pushed to XXX"
        match = re.search(r'Successfully pushed to ([^\s]+)', kaggle_output)
        if match:
            return match.group(1)

        # パターン2: kernel URL
        match = re.search(r'kaggle\.com/code/([^\s]+)', kaggle_output)
        if match:
            return match.group(1)

        # パターン3: kernel-metadata.jsonから読み取り（フォールバック）
        return None

    def get_kernel_id_from_metadata(self, notebook_path):
        """kernel-metadata.jsonからIDを読み取る"""
        metadata_file = Path(notebook_path) / 'kernel-metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                return metadata.get('id')
        return None

    def submit_to_kaggle(self, notebook_path, feature, attempt, auto_mode=True):
        """
        Kaggleにノートブックを提出し、自動監視を開始

        Args:
            notebook_path: Kaggle notebookのディレクトリパス
            feature: 特徴量名 (e.g., "real_rate")
            attempt: 試行番号
            auto_mode: True=完全自動化、False=通常モード（手動評価）

        Returns:
            bool: 提出成功ならTrue
        """
        print("=" * 70)
        print(f"[{datetime.now()}] >> Submitting to Kaggle")
        print("=" * 70)
        print(f"Feature: {feature}, Attempt: {attempt}")
        print(f"Notebook path: {notebook_path}")
        print("=" * 70)

        # 1. Kaggle提出
        try:
            # 環境変数を設定（UTF-8モード + 新しいAPI Token形式対応）
            env = os.environ.copy()
            env['PYTHONUTF8'] = '1'
            if 'KAGGLE_API_TOKEN' in env:
                # 新しいAPI Token形式の場合はそのまま使用
                pass

            result = subprocess.run(
                ['kaggle', 'kernels', 'push', '-p', notebook_path],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=self.project_root,
                env=env,
                encoding='utf-8',
                errors='replace'
            )

            print(f"\n[Kaggle Output]")
            print(result.stdout)

            if result.returncode != 0:
                print(f"\n[Kaggle Error]")
                print(result.stderr)
                print(f"\n[FAIL] Kaggle submission failed")
                return False

        except subprocess.TimeoutExpired:
            print(f"[FAIL] Kaggle submission timeout")
            return False
        except Exception as e:
            print(f"[FAIL] Error during Kaggle submission: {e}")
            return False

        # 2. Kernel IDを抽出
        kernel_id = self.extract_kernel_id(result.stdout)

        if not kernel_id:
            # フォールバック: metadata.jsonから読み取り
            kernel_id = self.get_kernel_id_from_metadata(notebook_path)

        if not kernel_id:
            print(f"[FAIL] Failed to extract kernel ID")
            print(f"Output: {result.stdout}")
            return False

        print(f"\n[OK] Kernel ID: {kernel_id}")

        # 3. state.json更新
        self.update_state({
            'status': 'waiting_training',
            'kaggle_kernel': kernel_id,
            'submitted_at': datetime.now().isoformat()
        })

        print(f"[OK] state.json updated")

        # 4. Git commit & push
        try:
            subprocess.run(['git', 'add', '-A'], cwd=self.project_root, check=True)
            subprocess.run(
                ['git', 'commit', '-m', f'kaggle: {feature} attempt {attempt} - submitted'],
                cwd=self.project_root,
                check=True
            )
            subprocess.run(['git', 'push'], cwd=self.project_root, check=True)
            print(f"[OK] Git committed and pushed")
        except subprocess.CalledProcessError as e:
            print(f"[WARN] Git operation failed: {e}")

        # 5. 監視スクリプト起動（auto_modeによって動作が変わる）
        if auto_mode:
            monitor_script = self.project_root / 'scripts' / 'auto_resume_after_kaggle.py'

            try:
                # Windowsの場合
                if sys.platform == 'win32':
                    subprocess.Popen(
                        ['python', str(monitor_script)],
                        creationflags=subprocess.CREATE_NEW_CONSOLE | subprocess.DETACHED_PROCESS,
                        cwd=self.project_root
                    )
                # Unix系の場合
                else:
                    subprocess.Popen(
                        ['python', str(monitor_script)],
                        start_new_session=True,
                        cwd=self.project_root
                    )

                print(f"[OK] Full-auto monitor started in background")

            except Exception as e:
                print(f"[WARN] Failed to start monitor (you can run it manually): {e}")
                print(f"Manual command: python {monitor_script}")

        # 6. ユーザーに通知
        print("\n" + "=" * 70)
        print("*** Kaggle Training Submitted Successfully!")
        print("=" * 70)
        print(f"Kernel URL: https://www.kaggle.com/code/{kernel_id}")

        if auto_mode:
            print(f"\n🤖 Full Auto Mode:")
            print(f"  - Monitor checks every 1 minute for up to 3 hours")
            print(f"  - Evaluator runs automatically when training completes")
            print(f"  - Next action is decided automatically (retry / next feature / done)")
            print(f"  - Error handling: auto-retry (network) / auto-skip (OOM)")
            print(f"\n You can now:")
            print(f"  - Close this terminal (monitoring continues in background)")
            print(f"  - Turn off your PC (monitoring stops, but Kaggle continues)")
            print(f"  - When monitor completes, check state.json for next action")
        else:
            print(f"\n👤 Manual Mode:")
            print(f"  - Training is running on Kaggle")
            print(f"  - When complete, run: 'Resume from where we left off'")
            print(f"  - Or manually fetch results and evaluate")

        print(f"\n Check Kaggle web UI for live training progress")
        print("=" * 70)

        return True

    def submit_and_exit(self, notebook_path, feature, attempt, auto_mode=True):
        """
        Kaggle提出 → 自動監視開始 → このセッションを終了

        Args:
            notebook_path: Kaggle notebookのディレクトリパス
            feature: 特徴量名
            attempt: 試行番号
            auto_mode: True=完全自動化、False=通常モード（手動評価）

        OrchestratorがKaggle提出後に呼び出す想定
        """
        success = self.submit_to_kaggle(notebook_path, feature, attempt, auto_mode=auto_mode)

        if success and auto_mode:
            print(f"\n🤖 Exiting orchestrator session (full-auto mode)...")
            print(f"(Auto-resume will handle the rest)")
            sys.exit(0)  # 正常終了
        elif success:
            print(f"\n👤 Submission complete (manual mode)")
            print(f"Say 'Resume from where we left off' when training completes")
            return True
        else:
            print(f"\n[FAIL] Submission failed. Staying in current session.")
            return False


def main():
    """テスト用エントリーポイント"""
    import argparse

    parser = argparse.ArgumentParser(description='Kaggle提出＋自動監視開始')
    parser.add_argument('notebook_path', help='Kaggle notebookディレクトリパス')
    parser.add_argument('feature', help='特徴量名 (e.g., real_rate)')
    parser.add_argument('attempt', type=int, help='試行番号')
    parser.add_argument('--no-exit', action='store_true', help='終了せずに戻る')
    parser.add_argument('--manual', action='store_true', help='手動モード（自動監視なし）')

    args = parser.parse_args()

    handler = KaggleSubmissionHandler()
    auto_mode = not args.manual

    if args.no_exit:
        success = handler.submit_to_kaggle(args.notebook_path, args.feature, args.attempt, auto_mode=auto_mode)
        sys.exit(0 if success else 1)
    else:
        handler.submit_and_exit(args.notebook_path, args.feature, args.attempt, auto_mode=auto_mode)


if __name__ == '__main__':
    main()
