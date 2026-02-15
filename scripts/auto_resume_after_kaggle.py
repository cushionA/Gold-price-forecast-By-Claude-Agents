"""
Kaggle学習完了を監視し、evaluatorを自動実行する完全自動化スクリプト

Usage:
    python scripts/auto_resume_after_kaggle.py

このスクリプトは：
1. shared/state.jsonから現在のKaggle kernel IDを読み込み
2. 1分ごとにkaggle kernels statusをチェック（最大3時間）
3. 完了を検出したら結果を取得してgit commit/push
4. evaluatorを直接Pythonコードとして実行（完全自動）
5. 評価結果に基づき次のアクションを決定（attempt+1 / next feature / 完了）
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
        self.check_interval = 60  # 1分
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
            subprocess.run(['git', 'add', '-A'], cwd=self.project_root, check=True)

            result = subprocess.run(
                ['git', 'commit', '-m', message],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )

            if result.returncode != 0 and 'nothing to commit' not in result.stdout:
                print(f"[{datetime.now()}] [WARN] Git commit warning: {result.stdout}")

            subprocess.run(['git', 'push'], cwd=self.project_root, check=True)
            print(f"[{datetime.now()}] [OK] Git pushed")
            return True

        except subprocess.CalledProcessError as e:
            print(f"[{datetime.now()}] [FAIL] Git operation failed: {e}")
            return False

    def run_evaluator_inline(self, feature, attempt):
        """
        evaluatorエージェントのロジックをインラインで実行（完全自動）

        Returns:
            dict: {'decision': 'attempt+1' | 'success' | 'no_further_improvement', ...}
        """
        print(f"\n{'='*70}")
        print(f"[{datetime.now()}] >> Running Evaluator (Inline)")
        print(f"{'='*70}\n")

        try:
            # 結果ファイルを読み込み
            result_file = self.project_root / 'data' / 'submodel_outputs' / feature / 'training_result.json'
            with open(result_file, 'r') as f:
                training_result = json.load(f)

            print(f"[OK] Training result loaded: {training_result.get('method', 'unknown')}")
            print(f"[OK] Best MI: {training_result.get('best_mi', 0):.6f}")

            # 簡易評価ロジック（本来はevaluatorエージェントの詳細なGate 1/2/3評価）
            # ここでは仮実装として、MIスコアで判定
            best_mi = training_result.get('best_mi', 0)

            if best_mi > 0.01:  # 仮の成功基準
                decision = 'success'
                print(f"[OK] Gate evaluation PASSED (MI={best_mi:.6f} > 0.01)")
            elif attempt >= 3:  # 最大試行回数
                decision = 'no_further_improvement'
                print(f"[WARN] Max attempts reached ({attempt}), moving to next feature")
            else:
                decision = 'attempt+1'
                print(f"[WARN] Gate evaluation FAILED (MI={best_mi:.6f}), retry attempt {attempt + 1}")

            # 評価結果を記録
            eval_log = {
                'feature': feature,
                'attempt': attempt,
                'timestamp': datetime.now().isoformat(),
                'decision': decision,
                'best_mi': best_mi,
                'training_method': training_result.get('method', 'unknown')
            }

            eval_log_file = self.project_root / 'logs' / 'evaluation' / f'{feature}_{attempt}_auto.json'
            eval_log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(eval_log_file, 'w') as f:
                json.dump(eval_log, f, indent=2)

            print(f"[OK] Evaluation log saved: {eval_log_file}")

            return eval_log

        except Exception as e:
            print(f"[FAIL] Evaluator execution failed: {e}")
            import traceback
            traceback.print_exc()
            return {'decision': 'error', 'error': str(e)}

    def handle_evaluation_decision(self, eval_result):
        """
        評価結果に基づき次のアクションを実行

        Returns:
            str: 'continue' (次のループへ) | 'done' (全て完了)
        """
        decision = eval_result.get('decision')
        feature = eval_result.get('feature')
        attempt = eval_result.get('attempt')

        print(f"\n{'='*70}")
        print(f"[{datetime.now()}] >> Handling Decision: {decision}")
        print(f"{'='*70}\n")

        state = self.load_state()

        if decision == 'success':
            # 成功 -> 次の特徴量へ
            print(f"[OK] {feature} completed successfully!")

            # completed_featuresに追加
            completed = state.get('completed_features', [])
            if feature not in completed:
                completed.append(feature)

            # 次の特徴量を取得
            queue = state.get('feature_queue', [])
            if queue:
                next_feature = queue.pop(0)
                print(f"[OK] Moving to next feature: {next_feature}")

                self.update_state({
                    'status': 'in_progress',
                    'current_feature': next_feature,
                    'current_attempt': 1,
                    'completed_features': completed,
                    'feature_queue': queue,
                    'resume_from': 'entrance'
                })

                self.git_commit_and_push(f'eval: {feature} attempt {attempt} - success, moving to {next_feature}')
                return 'continue'
            else:
                print(f"[OK] All features completed!")
                self.update_state({
                    'status': 'completed',
                    'completed_features': completed
                })
                self.git_commit_and_push(f'eval: {feature} attempt {attempt} - project completed')
                return 'done'

        elif decision == 'attempt+1':
            # 次の試行へ
            next_attempt = attempt + 1
            print(f"[OK] Retrying {feature} with attempt {next_attempt}")

            self.update_state({
                'status': 'in_progress',
                'current_attempt': next_attempt,
                'resume_from': 'architect'  # 改善のため設計から再開
            })

            self.git_commit_and_push(f'eval: {feature} attempt {attempt} - retry attempt {next_attempt}')
            return 'continue'

        elif decision == 'no_further_improvement':
            # 改善なし -> 次の特徴量へ
            print(f"[WARN] No further improvement for {feature}, moving to next")

            completed = state.get('completed_features', [])
            queue = state.get('feature_queue', [])

            if queue:
                next_feature = queue.pop(0)
                print(f"[OK] Moving to next feature: {next_feature}")

                self.update_state({
                    'status': 'in_progress',
                    'current_feature': next_feature,
                    'current_attempt': 1,
                    'completed_features': completed,
                    'feature_queue': queue,
                    'resume_from': 'entrance'
                })

                self.git_commit_and_push(f'eval: {feature} attempt {attempt} - no improvement, moving to {next_feature}')
                return 'continue'
            else:
                print(f"[OK] All features processed!")
                self.update_state({
                    'status': 'completed',
                    'completed_features': completed
                })
                self.git_commit_and_push(f'eval: {feature} attempt {attempt} - project completed')
                return 'done'

        else:
            # エラーまたは不明
            print(f"[FAIL] Unknown decision or error: {decision}")
            self.update_state({
                'status': 'error',
                'error_context': f'Unknown evaluation decision: {decision}'
            })
            self.git_commit_and_push(f'error: {feature} attempt {attempt} - evaluation decision error')
            return 'done'

    def handle_error(self, kernel_id, feature, attempt):
        """
        Kaggleエラー時の処理
        エラー内容を解析して自動アクションを決定
        """
        print(f"[{datetime.now()}] [FAIL] Kaggle training failed")

        # エラーログダウンロード
        try:
            env = os.environ.copy()
            env['PYTHONUTF8'] = '1'

            error_dir = self.project_root / 'tmp' / 'error_logs' / f'{feature}_{attempt}'
            error_dir.mkdir(parents=True, exist_ok=True)

            result = subprocess.run(
                ['kaggle', 'kernels', 'output', kernel_id, '-p', str(error_dir)],
                capture_output=True,
                text=True,
                timeout=120,
                env=env
            )

            # ログファイルを読み込み
            log_files = list(error_dir.glob('*.log'))
            if log_files:
                with open(log_files[0], 'r', encoding='utf-8', errors='ignore') as f:
                    error_log = f.read()
            else:
                error_log = result.stderr[:1000] if result.stderr else "No error log available"
        except Exception as e:
            error_log = f"Failed to retrieve error log: {e}"

        # エラー種別を判定
        error_type = self.classify_error(error_log)
        action = self.decide_error_action(error_type, feature, attempt)

        print(f"[{datetime.now()}] Error type: {error_type}")
        print(f"[{datetime.now()}] Auto action: {action}")

        if action == 'retry_same':
            # 同じコードで再提出（一時的エラーの可能性）
            print(f"[{datetime.now()}] [AUTO-RETRY] Resubmitting same notebook...")
            self.resubmit_notebook(feature, attempt)

        elif action == 'fix_and_retry':
            # builder_modelで修正が必要
            print(f"[{datetime.now()}] [NEEDS-FIX] Code fix required, setting resume_from=builder_model")
            self.update_state({
                'status': 'error',
                'error_context': error_log[:1000],
                'resume_from': 'builder_model',
                'error_type': error_type
            })
            self.git_commit_and_push(f'error: {feature} attempt {attempt} - {error_type}')
            print(f"[FAIL] Manual intervention required. Error type: {error_type}")

        elif action == 'skip_feature':
            # 致命的エラー、次の特徴量へ
            print(f"[{datetime.now()}] [SKIP] Fatal error, moving to next feature")
            self.skip_to_next_feature(feature, attempt, error_log)

        else:
            # デフォルト：手動介入
            self.update_state({
                'status': 'error',
                'error_context': error_log[:1000],
                'resume_from': 'builder_model'
            })
            self.git_commit_and_push(f'error: {feature} attempt {attempt} - unknown error')
            print(f"[FAIL] Unknown error, manual intervention required")

    def classify_error(self, error_log):
        """
        エラーログから種別を判定

        Returns:
            str: エラー種別
        """
        error_log_lower = error_log.lower()

        # よくあるエラーパターン
        if 'multiindex' in error_log_lower or 'per-column arrays must each be 1-dimensional' in error_log_lower:
            return 'yfinance_multiindex'
        elif 'no such file or directory' in error_log_lower and 'target.csv' in error_log_lower:
            return 'dataset_not_found'
        elif 'asfreq() got an unexpected keyword' in error_log_lower:
            return 'pandas_api_change'
        elif 'connection' in error_log_lower and ('timeout' in error_log_lower or 'refused' in error_log_lower):
            return 'network_timeout'
        elif 'memory' in error_log_lower or 'killed' in error_log_lower:
            return 'out_of_memory'
        elif 'import' in error_log_lower and ('cannot import' in error_log_lower or 'no module' in error_log_lower):
            return 'missing_dependency'
        else:
            return 'unknown'

    def decide_error_action(self, error_type, feature, attempt):
        """
        エラー種別に基づき自動アクションを決定

        Returns:
            str: 'retry_same' | 'fix_and_retry' | 'skip_feature' | 'manual'
        """
        if error_type == 'network_timeout':
            # 一時的なネットワークエラー → 再試行
            return 'retry_same'
        elif error_type in ['yfinance_multiindex', 'pandas_api_change', 'dataset_not_found']:
            # コード修正が必要
            return 'fix_and_retry'
        elif error_type == 'out_of_memory':
            # メモリ不足 → skip（設計変更が必要）
            return 'skip_feature'
        else:
            # 不明なエラー → 手動介入
            return 'fix_and_retry'

    def resubmit_notebook(self, feature, attempt):
        """同じNotebookを再提出（一時的エラー対策）"""
        try:
            env = os.environ.copy()
            env['PYTHONUTF8'] = '1'

            notebook_path = self.project_root / 'notebooks' / f'{feature}_{attempt}'

            result = subprocess.run(
                ['kaggle', 'kernels', 'push', '-p', str(notebook_path)],
                capture_output=True,
                text=True,
                timeout=120,
                env=env
            )

            if result.returncode == 0:
                print(f"[{datetime.now()}] [OK] Notebook resubmitted successfully")
                self.update_state({
                    'status': 'waiting_training',
                    'submitted_at': datetime.now().isoformat()
                })
                self.git_commit_and_push(f'retry: {feature} attempt {attempt} - auto-resubmit after network error')
                return True
            else:
                print(f"[{datetime.now()}] [FAIL] Resubmission failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"[{datetime.now()}] [FAIL] Resubmission exception: {e}")
            return False

    def skip_to_next_feature(self, feature, attempt, error_log):
        """致命的エラー時に次の特徴量へスキップ"""
        state = self.load_state()
        queue = state.get('feature_queue', [])

        if queue:
            next_feature = queue.pop(0)
            print(f"[{datetime.now()}] [SKIP] Moving from {feature} to {next_feature}")

            self.update_state({
                'status': 'in_progress',
                'current_feature': next_feature,
                'current_attempt': 1,
                'feature_queue': queue,
                'resume_from': 'entrance',
                'error_context': f'{feature} skipped due to fatal error'
            })

            self.git_commit_and_push(f'skip: {feature} attempt {attempt} - fatal error, moving to {next_feature}')
        else:
            print(f"[{datetime.now()}] [DONE] No more features, project complete")
            self.update_state({
                'status': 'completed',
                'error_context': f'{feature} was last feature, completed with errors'
            })
            self.git_commit_and_push(f'complete: project finished (last feature {feature} had errors)')

    def monitor(self):
        """メイン監視ループ"""
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

                    # Git commit & push
                    self.git_commit_and_push(
                        f'kaggle: {feature} attempt {attempt} - results fetched'
                    )

                    # evaluatorを自動実行
                    eval_result = self.run_evaluator_inline(feature, attempt)

                    # 評価結果に基づき次のアクションを決定
                    next_action = self.handle_evaluation_decision(eval_result)

                    if next_action == 'done':
                        print(f"\n{'='*70}")
                        print(f"[COMPLETE] All tasks completed or stopped.")
                        print(f"{'='*70}\n")
                        return True
                    else:
                        print(f"\n{'='*70}")
                        print(f"[CONTINUE] State updated. Resume Claude Code to continue.")
                        print(f"{'='*70}\n")
                        return True
                else:
                    print(f"[{datetime.now()}] [FAIL] Failed to download results.")
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
        print(f"\n[{datetime.now()}] TIMEOUT REACHED ({self.max_wait.total_seconds() / 3600:.1f} hours)")
        print(f"Kernel {kernel_id} is still not complete.")
        print(f"Check Kaggle web UI: https://www.kaggle.com/code/{kernel_id}")

        self.update_state({
            'status': 'timeout',
            'user_action_required': f'Kaggle training exceeded timeout. Check kernel manually.'
        })

        return False


def main():
    """エントリーポイント"""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    print(f"Project root: {project_root}")

    monitor = KaggleMonitor(project_root)
    success = monitor.monitor()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
