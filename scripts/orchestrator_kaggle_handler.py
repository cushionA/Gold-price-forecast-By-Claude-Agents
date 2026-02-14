"""
Orchestrator用のKaggle統合ハンドラー

Kaggle提出 → 自動監視スクリプト起動 → セッション終了の一連の処理を提供
"""

import subprocess
import sys
import json
import re
from datetime import datetime
from pathlib import Path


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

    def submit_to_kaggle(self, notebook_path, feature, attempt):
        """
        Kaggleにノートブックを提出し、自動監視を開始

        Args:
            notebook_path: Kaggle notebookのディレクトリパス
            feature: 特徴量名 (e.g., "real_rate")
            attempt: 試行番号

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
            result = subprocess.run(
                ['kaggle', 'kernels', 'push', '-p', notebook_path],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=self.project_root
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

        # 5. 自動監視スクリプトをバックグラウンド起動
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

            print(f"[OK] Auto-resume monitor started in background")

        except Exception as e:
            print(f"[WARN] Failed to start monitor (you can run it manually): {e}")
            print(f"Manual command: python {monitor_script}")

        # 6. ユーザーに通知
        print("\n" + "=" * 70)
        print("*** Kaggle Training Submitted Successfully!")
        print("=" * 70)
        print(f"Kernel URL: https://www.kaggle.com/code/{kernel_id}")
        print(f"\n Monitoring:")
        print(f"  - Auto-resume script is running in the background")
        print(f"  - It will check every 5 minutes for up to 3 hours")
        print(f"  - Claude Code will automatically restart when training completes")
        print(f"\n You can now:")
        print(f"  - Close this terminal (monitoring continues in background)")
        print(f"  - Turn off your PC (monitoring will stop, but Kaggle continues)")
        print(f"  - Check Kaggle web UI for live training progress")
        print("=" * 70)

        return True

    def submit_and_exit(self, notebook_path, feature, attempt):
        """
        Kaggle提出 → 自動監視開始 → このセッションを終了

        OrchestratorがKaggle提出後に呼び出す想定
        """
        success = self.submit_to_kaggle(notebook_path, feature, attempt)

        if success:
            print(f"\nSTOP Exiting orchestrator session...")
            print(f"(Auto-resume will handle the rest)")
            sys.exit(0)  # 正常終了
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

    args = parser.parse_args()

    handler = KaggleSubmissionHandler()

    if args.no_exit:
        success = handler.submit_to_kaggle(args.notebook_path, args.feature, args.attempt)
        sys.exit(0 if success else 1)
    else:
        handler.submit_and_exit(args.notebook_path, args.feature, args.attempt)


if __name__ == '__main__':
    main()
