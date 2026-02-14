"""
Kaggle学習完了を監視し、Claude Code CLIを自動起動するスクリプト

Usage:
    python scripts/auto_resume_after_kaggle.py

このスクリプトは：
1. shared/state.jsonから現在のKaggle kernel IDを読み込み
2. 5分ごとにkaggle kernels statusをチェック（最大3時間）
3. 完了を検出したら結果を取得してgit commit/push
4. Claude Code CLIを自動再起動して評価を続行
"""

import subprocess
import time
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path


class KaggleMonitor:
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.state_file = self.project_root / 'shared' / 'state.json'
        self.check_interval = 300  # 5分
        self.max_wait = timedelta(hours=3)

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

    def check_kaggle_status(self, kernel_id):
        """
        Kaggle kernel statusをチェック

        Returns:
            str: 'complete', 'running', 'error', 'unknown'
        """
        try:
            # 環境変数を設定（UTF-8モード）
            env = os.environ.copy()
            env['PYTHONUTF8'] = '1'

            result = subprocess.run(
                ['kaggle', 'kernels', 'status', kernel_id],
                capture_output=True,
                text=True,
                timeout=30,
                env=env
            )

            output = result.stdout.lower()

            if 'complete' in output:
                return 'complete'
            elif 'error' in output or 'failed' in output:
                return 'error'
            elif 'running' in output:
                return 'running'
            else:
                return 'unknown'

        except subprocess.TimeoutExpired:
            print(f"[{datetime.now()}] [WARN] Kaggle API timeout")
            return 'unknown'
        except Exception as e:
            print(f"[{datetime.now()}] [WARN] Error checking status: {e}")
            return 'unknown'

    def fetch_kaggle_results(self, kernel_id, output_dir):
        """Kaggle結果を取得"""
        os.makedirs(output_dir, exist_ok=True)

        try:
            # 環境変数を設定（UTF-8モード）
            env = os.environ.copy()
            env['PYTHONUTF8'] = '1'

            result = subprocess.run(
                ['kaggle', 'kernels', 'output', kernel_id, '-p', output_dir],
                capture_output=True,
                text=True,
                timeout=120,
                env=env
            )

            if result.returncode == 0:
                print(f"[{datetime.now()}] [OK] Results downloaded to {output_dir}")
                return True
            else:
                print(f"[{datetime.now()}] [FAIL] Download failed: {result.stderr}")
                return False

        except Exception as e:
            print(f"[{datetime.now()}] [FAIL] Error downloading results: {e}")
            return False

    def git_commit_and_push(self, message):
        """Git commit & push"""
        try:
            # Git add
            subprocess.run(['git', 'add', '-A'], cwd=self.project_root, check=True)

            # Git commit
            result = subprocess.run(
                ['git', 'commit', '-m', message],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )

            if result.returncode != 0 and 'nothing to commit' not in result.stdout:
                print(f"[{datetime.now()}] [WARN] Git commit warning: {result.stdout}")

            # Git push
            subprocess.run(['git', 'push'], cwd=self.project_root, check=True)
            print(f"[{datetime.now()}] [OK] Git pushed")
            return True

        except subprocess.CalledProcessError as e:
            print(f"[{datetime.now()}] [FAIL] Git operation failed: {e}")
            return False

    def resume_claude_code(self, message):
        """Claude Code CLIを再起動"""
        try:
            print(f"[{datetime.now()}] >> Resuming Claude Code...")

            # Claude Code CLIを起動（このプロセスは新しいセッションで実行される）
            subprocess.Popen(
                ['claude-code', '--message', message, '--project', str(self.project_root)],
                start_new_session=True
            )

            print(f"[{datetime.now()}] [OK] Claude Code launched")
            return True

        except Exception as e:
            print(f"[{datetime.now()}] [FAIL] Failed to launch Claude Code: {e}")
            return False

    def handle_error(self, kernel_id, feature, attempt):
        """Kaggleエラー時の処理"""
        print(f"[{datetime.now()}] [FAIL] Kaggle training failed")

        # エラーログ取得を試みる
        try:
            result = subprocess.run(
                ['kaggle', 'kernels', 'output', kernel_id, '-p', 'tmp/error_logs/'],
                capture_output=True,
                text=True,
                timeout=120
            )
            error_log = result.stderr[:1000] if result.stderr else "No error log available"
        except:
            error_log = "Failed to retrieve error log"

        # state.jsonに記録
        self.update_state({
            'status': 'error',
            'error_context': error_log,
            'resume_from': 'builder_model'
        })

        # Git commit
        self.git_commit_and_push(f'error: {feature} attempt {attempt} - kaggle training failed')

        # Claude Code再開（エラー修正を促す）
        self.resume_claude_code(
            f"Kaggle training for {feature} attempt {attempt} failed. "
            f"Error: {error_log[:200]}. Please review the error and regenerate the training script."
        )

    def monitor(self):
        """メイン監視ループ"""
        # state.jsonから情報を読み込み
        state = self.load_state()

        if state.get('status') != 'waiting_training':
            print(f"[{datetime.now()}] [FAIL] Error: state.status is not 'waiting_training'")
            print(f"Current status: {state.get('status')}")
            sys.exit(1)

        kernel_id = state.get('kaggle_kernel')
        feature = state.get('current_feature')
        attempt = state.get('current_attempt')

        if not kernel_id:
            print(f"[{datetime.now()}] [FAIL] Error: No kaggle_kernel in state.json")
            sys.exit(1)

        print("=" * 70)
        print(f"[{datetime.now()}] >> Kaggle Training Monitor Started")
        print("=" * 70)
        print(f"Kernel ID: {kernel_id}")
        print(f"Feature: {feature}, Attempt: {attempt}")
        print(f"Check interval: {self.check_interval}s (5 minutes)")
        print(f"Max wait time: {self.max_wait.total_seconds() / 3600:.1f} hours")
        print("=" * 70)

        start_time = datetime.now()
        check_count = 0

        while datetime.now() - start_time < self.max_wait:
            check_count += 1
            elapsed = datetime.now() - start_time
            elapsed_str = f"{elapsed.total_seconds() / 60:.1f} min"

            print(f"\n[{datetime.now()}] Check #{check_count} (elapsed: {elapsed_str})")

            status = self.check_kaggle_status(kernel_id)

            if status == 'complete':
                print(f"[{datetime.now()}] [OK][OK][OK] Training COMPLETE! [OK][OK][OK]")

                # 結果を取得
                output_dir = self.project_root / 'data' / 'submodel_outputs' / feature
                if self.fetch_kaggle_results(kernel_id, str(output_dir)):

                    # state.json更新
                    self.update_state({
                        'status': 'in_progress',
                        'resume_from': 'evaluator'
                    })

                    # Git commit & push
                    self.git_commit_and_push(
                        f'kaggle: {feature} attempt {attempt} - results fetched'
                    )

                    # Claude Code再開
                    self.resume_claude_code(
                        f"Kaggle training for {feature} attempt {attempt} is complete. "
                        f"Results have been downloaded. Please run the evaluator to assess performance."
                    )

                    print(f"[{datetime.now()}] *** All done! Claude Code will resume evaluation.")
                    return True
                else:
                    print(f"[{datetime.now()}] [FAIL] Failed to download results. Manual intervention required.")
                    return False

            elif status == 'error':
                self.handle_error(kernel_id, feature, attempt)
                return False

            elif status == 'running':
                print(f"[{datetime.now()}] ... Still running... (next check in {self.check_interval}s)")

            else:
                print(f"[{datetime.now()}] [WARN] Unknown status. Will check again.")

            time.sleep(self.check_interval)

        # タイムアウト
        print(f"\n[{datetime.now()}] ⏰ TIMEOUT REACHED ({self.max_wait.total_seconds() / 3600:.1f} hours)")
        print(f"Kernel {kernel_id} is still not complete.")
        print(f"Manual intervention required. Check Kaggle web UI:")
        print(f"https://www.kaggle.com/code/{kernel_id}")

        # state.jsonに記録
        self.update_state({
            'status': 'timeout',
            'user_action_required': f'Kaggle training exceeded {self.max_wait.total_seconds() / 3600:.1f}h timeout. Check kernel manually.'
        })

        return False


def main():
    """エントリーポイント"""
    # プロジェクトルートを検出
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    print(f"Project root: {project_root}")

    # 監視開始
    monitor = KaggleMonitor(project_root)
    success = monitor.monitor()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
