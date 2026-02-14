"""
評価完了後の自動クリーン＋再開スクリプト

Evaluatorが評価完了後に呼び出し、コンテキストをクリアして次の試行を開始
"""

import subprocess
import sys
import json
from datetime import datetime
from pathlib import Path


class AutoCleanResume:
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.state_file = self.project_root / 'shared' / 'state.json'

    def load_state(self):
        """state.jsonを読み込み"""
        with open(self.state_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def git_commit_and_push(self, message):
        """Git commit & push"""
        try:
            subprocess.run(['git', 'add', '-A'], cwd=self.project_root, check=True)

            result = subprocess.run(
                ['git', 'commit', '-m', message],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )

            if result.returncode != 0 and 'nothing to commit' not in result.stdout:
                print(f"[WARN] Git commit warning: {result.stdout}")

            subprocess.run(['git', 'push'], cwd=self.project_root, check=True)
            print(f"[OK] Git pushed: {message}")
            return True

        except subprocess.CalledProcessError as e:
            print(f"[FAIL] Git operation failed: {e}")
            return False

    def clean_context(self):
        """
        Claude Codeのコンテキストをクリア

        Note: 'claude clean' コマンドはCLIバージョンによって異なる可能性あり
        """
        try:
            # Attempt 1: claude clean
            result = subprocess.run(
                ['claude', 'clean'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                print(f"[OK] Context cleaned (claude clean)")
                return True

        except subprocess.CalledProcessError:
            pass
        except FileNotFoundError:
            pass

        try:
            # Attempt 2: claude-code clean
            result = subprocess.run(
                ['claude-code', 'clean'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                print(f"[OK] Context cleaned (claude-code clean)")
                return True

        except subprocess.CalledProcessError:
            pass
        except FileNotFoundError:
            pass

        print(f"[WARN] Failed to clean context (command not found or failed)")
        print(f"   This is optional - continuing anyway.")
        return False

    def resume_with_fresh_context(self, message):
        """
        新しいコンテキストでClaude Code CLIを再起動

        Args:
            message: 再開時のメッセージ
        """
        try:
            print(f"\n>> Resuming Claude Code with fresh context...")

            # Claude Code CLIを新しいセッションで起動
            if sys.platform == 'win32':
                subprocess.Popen(
                    ['claude-code', '--message', message, '--project', str(self.project_root)],
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:
                subprocess.Popen(
                    ['claude-code', '--message', message, '--project', str(self.project_root)],
                    start_new_session=True
                )

            print(f"[OK] Claude Code launched in new session")
            return True

        except Exception as e:
            print(f"[FAIL] Failed to launch Claude Code: {e}")
            return False

    def execute(self, feature, attempt, decision):
        """
        評価完了後の一連の処理を実行

        Args:
            feature: 特徴量名
            attempt: 試行番号
            decision: 評価決定 ('attempt+1', 'no_further_improvement', 'success')

        Returns:
            bool: 成功ならTrue
        """
        print("=" * 70)
        print(f"[{datetime.now()}]  Auto Clean & Resume")
        print("=" * 70)
        print(f"Feature: {feature}, Attempt: {attempt}")
        print(f"Decision: {decision}")
        print("=" * 70)

        # 1. Git commit & push（評価結果を保存）
        commit_msg = f"eval: {feature} attempt {attempt} - {decision}"
        if not self.git_commit_and_push(commit_msg):
            print(f"[WARN] Git push failed, but continuing...")

        # 2. コンテキストクリーン（オプショナル）
        print(f"\n Cleaning context...")
        self.clean_context()

        # 3. 次のアクションを決定
        state = self.load_state()
        next_feature = state.get('current_feature')
        next_attempt = state.get('current_attempt')
        resume_from = state.get('resume_from')

        if decision == 'attempt+1':
            resume_message = (
                f"Continuing {feature} with attempt {next_attempt}. "
                f"Previous attempt {attempt} completed evaluation. "
                f"Please proceed with the next iteration based on the improvement plan."
            )
        elif decision == 'no_further_improvement':
            resume_message = (
                f"{feature} development complete after {attempt} attempts. "
                f"Moving to next feature: {next_feature}. "
                f"Please start from {resume_from} agent."
            )
        elif decision == 'success':
            resume_message = (
                f"{feature} attempt {attempt} passed all gates! "
                f"Moving to next feature: {next_feature}. "
                f"Please start from {resume_from} agent."
            )
        else:
            resume_message = "Resume from where we left off"

        # 4. 新しいコンテキストで再開
        print(f"\n📋 Next action: {resume_from} for {next_feature} attempt {next_attempt}")
        print(f"Resume message: {resume_message[:100]}...")

        success = self.resume_with_fresh_context(resume_message)

        if success:
            print("\n" + "=" * 70)
            print("[OK] Auto Clean & Resume Complete!")
            print("=" * 70)
            print(f"  - Context cleaned")
            print(f"  - Git pushed")
            print(f"  - Claude Code restarted with fresh context")
            print(f"  - Next: {resume_from} for {next_feature}")
            print("=" * 70)
        else:
            print("\n" + "=" * 70)
            print("[WARN] Auto Resume Failed")
            print("=" * 70)
            print(f"  - You can manually restart with:")
            print(f"    claude-code --message 'Resume from where we left off'")
            print("=" * 70)

        return success

    def execute_and_exit(self, feature, attempt, decision):
        """
        評価完了後の処理を実行し、このセッションを終了

        Evaluatorが評価完了後に呼び出す想定
        """
        success = self.execute(feature, attempt, decision)

        print(f"\nSTOP Exiting current session...")
        print(f"(New Claude Code session has been started)")

        sys.exit(0)  # 正常終了（新しいセッションが起動済み）


def main():
    """テスト用エントリーポイント"""
    import argparse

    parser = argparse.ArgumentParser(description='評価後の自動クリーン＋再開')
    parser.add_argument('feature', help='特徴量名 (e.g., real_rate)')
    parser.add_argument('attempt', type=int, help='試行番号')
    parser.add_argument('decision', help='評価決定 (attempt+1, no_further_improvement, success)')
    parser.add_argument('--no-exit', action='store_true', help='終了せずに戻る')

    args = parser.parse_args()

    handler = AutoCleanResume()

    if args.no_exit:
        success = handler.execute(args.feature, args.attempt, args.decision)
        sys.exit(0 if success else 1)
    else:
        handler.execute_and_exit(args.feature, args.attempt, args.decision)


if __name__ == '__main__':
    main()
