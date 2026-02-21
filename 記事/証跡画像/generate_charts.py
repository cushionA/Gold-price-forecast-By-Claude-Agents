"""
技術記事用チャート生成スクリプト (v2 - draw.ioスタイル準拠)
.drawioファイルのデザインをmatplotlibで再現
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#FAFAFA'
plt.rcParams['axes.edgecolor'] = '#DDDDDD'
plt.rcParams['axes.grid'] = False
plt.rcParams['xtick.color'] = '#666666'
plt.rcParams['ytick.color'] = '#666666'

output_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# draw.io カラーパレット
# ============================================================
C_RED = '#e74c3c'
C_RED_DARK = '#b85450'
C_GREEN = '#2ecc71'
C_GREEN_DARK = '#27ae60'
C_GREEN_BORDER = '#1a7a3e'
C_BLUE = '#3498db'
C_BLUE_DARK = '#2980b9'
C_ORANGE = '#e67e22'
C_PURPLE = '#9b59b6'
C_PURPLE_DARK = '#8e44ad'
C_YELLOW = '#f1c40f'
C_TEAL = '#1abc9c'
C_GRAY = '#95a5a6'
C_GRAY_LIGHT = '#bdc3c7'
C_GRAY_DARK = '#7f8c8d'
C_DARK = '#34495e'
C_BG_LIGHT = '#FAFAFA'
C_BG_PANEL = '#F5F5F5'


def add_rounded_box(ax, x, y, w, h, text, color, sublabel=None, fontsize=11, alpha=0.12):
    """draw.ioスタイルの角丸ボックスを描画"""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                          facecolor=color, alpha=alpha, edgecolor=color,
                          linewidth=2.2, zorder=2)
    ax.add_patch(box)
    if sublabel:
        ax.text(x + w/2, y + h*0.62, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=color, zorder=3)
        ax.text(x + w/2, y + h*0.25, sublabel, ha='center', va='center',
                fontsize=fontsize-2, color='#555555', zorder=3)
    else:
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=color, zorder=3)
    return box


def draw_arrow(ax, start, end, color='#555555', lw=2, style='->', dashed=False, connectionstyle=None):
    """draw.ioスタイルの矢印を描画"""
    ls = '--' if dashed else '-'
    props = dict(arrowstyle=style, color=color, lw=lw, linestyle=ls)
    if connectionstyle:
        props['connectionstyle'] = connectionstyle
    ax.annotate('', xy=end, xytext=start, arrowprops=props, zorder=1)


# ============================================================
# Chart 1: パイプライン全体像（フロー図）- draw.io準拠
# ============================================================
fig, ax = plt.subplots(figsize=(18, 10))
ax.set_xlim(-0.5, 17)
ax.set_ylim(-0.5, 10.5)
ax.axis('off')
fig.patch.set_facecolor('white')

# Title
ax.text(8.25, 10.0, '自動開発パイプライン全体像', ha='center', va='center',
        fontsize=22, fontweight='bold', color=C_DARK)

# ===== Phase labels =====
ax.text(4.5, 9.2, '── 設計フェーズ ──', ha='center', fontsize=11, color=C_GRAY, style='italic')
ax.text(4.5, 6.2, '── 構築フェーズ ──', ha='center', fontsize=11, color=C_GRAY, style='italic')

# ===== TOP ROW: Design Phase =====
add_rounded_box(ax, 0.3, 7.5, 2.8, 1.5, 'entrance\n(Opus)', C_RED, '要件定義')
add_rounded_box(ax, 3.6, 7.5, 2.8, 1.5, 'researcher\n(Sonnet)', C_ORANGE, '手法調査')
add_rounded_box(ax, 6.9, 7.5, 2.8, 1.5, 'architect\n(Opus)', '#c0942f', 'ファクトチェック → 設計書')

# ===== BOTTOM ROW: Build Phase =====
add_rounded_box(ax, 0.3, 4.5, 2.8, 1.5, 'builder_data\n(Sonnet)', C_GREEN_DARK, 'データ取得・前処理')
add_rounded_box(ax, 3.6, 4.5, 2.8, 1.5, 'datachecker\n(Haiku)', C_TEAL, '品質7ステップチェック')
add_rounded_box(ax, 6.9, 4.5, 2.8, 1.5, 'builder_model\n(Sonnet)', C_BLUE, 'PyTorchスクリプト生成')

# ===== RIGHT COLUMN: Cloud + Evaluation =====
add_rounded_box(ax, 10.5, 4.5, 3.0, 1.5, 'Kaggle Cloud\nGPU', C_PURPLE, 'クラウド学習')
add_rounded_box(ax, 10.5, 7.5, 3.0, 1.5, 'evaluator\n(Opus)', C_RED, 'Gate 1/2/3 評価')

# ===== PASS box =====
add_rounded_box(ax, 14.2, 7.6, 2.3, 1.2, 'PASS\n次のサブモデルへ', C_GREEN_DARK, alpha=0.15)

# ===== Orchestrator =====
box_orch = FancyBboxPatch((14.2, 4.5), 2.5, 1.5, boxstyle="round,pad=0.15",
                           facecolor=C_DARK, alpha=0.08, edgecolor=C_DARK,
                           linewidth=2, linestyle='--', zorder=2)
ax.add_patch(box_orch)
ax.text(15.45, 5.55, 'Orchestrator', ha='center', va='center',
        fontsize=11, fontweight='bold', color=C_DARK)
ax.text(15.45, 5.05, 'state.json管理\ngit自動コミット\nKaggle API操作', ha='center', va='center',
        fontsize=8, color='#555555')

# ===== validate_notebook annotation =====
note_box = FancyBboxPatch((7.1, 3.2), 2.4, 0.9, boxstyle="round,pad=0.1",
                           facecolor='#FFF9C4', alpha=0.9, edgecolor='#d6b656',
                           linewidth=1.2, zorder=2)
ax.add_patch(note_box)
ax.text(8.3, 3.65, 'validate_notebook.py\n(10 checks)', ha='center', va='center',
        fontsize=8, color='#8a7a2e')
draw_arrow(ax, (8.3, 4.1), (8.3, 4.5), color='#d6b656', lw=1.2, dashed=True)

# ===== PC off banner =====
pc_box = FancyBboxPatch((10.2, 3.2), 3.6, 0.9, boxstyle="round,pad=0.1",
                          facecolor=C_PURPLE, alpha=0.08, edgecolor=C_PURPLE,
                          linewidth=1.5, linestyle='--', zorder=2)
ax.add_patch(pc_box)
ax.text(12.0, 3.65, '[PC OFF OK] 学習継続', ha='center', va='center',
        fontsize=11, fontweight='bold', color=C_PURPLE)

# ===== ARROWS: Main flow =====
# entrance → researcher
draw_arrow(ax, (3.1, 8.25), (3.6, 8.25))
# researcher → architect
draw_arrow(ax, (6.4, 8.25), (6.9, 8.25))
# architect → builder_data (diagonal down)
draw_arrow(ax, (8.3, 7.5), (1.7, 6.0), connectionstyle='arc3,rad=0.25')
# builder_data → datachecker
draw_arrow(ax, (3.1, 5.25), (3.6, 5.25))
# datachecker → builder_model
draw_arrow(ax, (6.4, 5.25), (6.9, 5.25))
# builder_model → Kaggle
draw_arrow(ax, (9.7, 5.25), (10.5, 5.25), color=C_PURPLE, lw=2.5)
ax.text(10.1, 5.55, 'API', fontsize=8, color=C_PURPLE, ha='center')
# Kaggle → evaluator (up)
draw_arrow(ax, (12.0, 6.0), (12.0, 7.5), color=C_PURPLE, lw=2.5)
ax.text(12.35, 6.75, '結果取得', fontsize=8, color=C_PURPLE, ha='left', rotation=90)
# evaluator → PASS
draw_arrow(ax, (13.5, 8.2), (14.2, 8.2), color=C_GREEN_DARK, lw=2)

# ===== FAIL loop (evaluator → entrance) =====
draw_arrow(ax, (12.0, 9.0), (1.7, 9.0), color=C_RED, lw=2, dashed=True,
           connectionstyle='arc3,rad=-0.05')
# Draw vertical connectors for the loop
draw_arrow(ax, (12.0, 9.0), (12.0, 9.0), color=C_RED, lw=2)
ax.annotate('', xy=(1.7, 9.0), xytext=(12.0, 9.0),
            arrowprops=dict(arrowstyle='->', color=C_RED, lw=2, linestyle='--'))
ax.text(6.8, 9.35, 'FAIL → 改善ループ', fontsize=12, color=C_RED,
        fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor=C_RED, linewidth=1))

# ===== REJECT loop (datachecker → builder_data) =====
draw_arrow(ax, (5.0, 4.5), (1.7, 4.5), color=C_TEAL, lw=1.5, dashed=True,
           connectionstyle='arc3,rad=0.4')
ax.text(3.35, 3.8, 'REJECT（attempt消費なし）', fontsize=8, color=C_TEAL, ha='center',
        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.8, edgecolor=C_TEAL, linewidth=0.8))

# ===== Legend =====
legend_y = 0.3
legend_box = FancyBboxPatch((-0.2, -0.2), 12, 1.2, boxstyle="round,pad=0.1",
                              facecolor='#F8F8F8', alpha=0.9, edgecolor='#CCCCCC',
                              linewidth=1, zorder=2)
ax.add_patch(legend_box)

ax.plot([0.2, 0.8], [legend_y, legend_y], '-', color='#555555', lw=2)
ax.text(1.0, legend_y, '通常フロー', fontsize=10, color='#555555', va='center')

ax.plot([2.5, 3.1], [legend_y, legend_y], '--', color=C_RED, lw=2)
ax.text(3.3, legend_y, 'FAILループ', fontsize=10, color=C_RED, va='center')

ax.plot([5.0, 5.6], [legend_y, legend_y], '--', color=C_TEAL, lw=2)
ax.text(5.8, legend_y, 'REJECTループ', fontsize=10, color=C_TEAL, va='center')

ax.plot([8.0, 8.6], [legend_y, legend_y], '-', color=C_PURPLE, lw=2.5)
ax.text(8.8, legend_y, 'Kaggle API', fontsize=10, color=C_PURPLE, va='center')

ax.text(0.2, -0.05, '各エージェント完了後にOrchestratorがgit commit & push', fontsize=9, color='#888888')

plt.savefig(os.path.join(output_dir, 'pipeline_overview.png'), dpi=180, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Chart 1 saved: pipeline_overview.png")


# ============================================================
# Chart 2: サブモデルGate 3改善効果 - draw.ioスタイル
# ============================================================
fig, ax = plt.subplots(figsize=(15, 8))
fig.patch.set_facecolor('white')

submodels = [
    '実質金利', 'オプション市場', 'テクニカル', 'イールドカーブ',
    'ETFフロー', 'インフレ期待', '時間文脈', 'DXY',
    'クロスアセット', 'VIX', 'レジーム分類', 'CNY需要'
]
da_delta = [-0.57, -0.24, 0.05, 0.20, 0.45, 0.57, 0.58, 0.73, 0.76, 0.96, 1.34, 1.53]
sharpe_delta = [-0.249, -0.141, -0.092, -0.089, 0.377, 0.152, 0.113, 0.255, 0.040, 0.289, 0.377, 0.217]
attempts_sub = [6, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2]
status = ['FAIL', 'PASS', 'PASS', 'PASS', 'PASS', 'PASS', 'PASS', 'PASS', 'PASS', 'PASS', 'PASS', 'PASS']

y = np.arange(len(submodels))
bar_height = 0.32

# DA delta bars
for i, (d, s) in enumerate(zip(da_delta, status)):
    color = C_RED if d < 0 else C_BLUE
    alpha = 0.5 if s == 'FAIL' else 0.82
    ax.barh(i + bar_height/2, d, bar_height, color=color, alpha=alpha, edgecolor='white', linewidth=0.5, zorder=3)

# Sharpe delta bars
for i, (sh, s) in enumerate(zip(sharpe_delta, status)):
    color = '#c0392b' if sh < 0 else C_ORANGE
    alpha = 0.5 if s == 'FAIL' else 0.82
    ax.barh(i - bar_height/2, sh, bar_height, color=color, alpha=alpha, edgecolor='white', linewidth=0.5, zorder=3)

# Zero line
ax.axvline(x=0, color='#999999', linewidth=1, zorder=2)

# Y labels with background
ax.set_yticks(y)
ax.set_yticklabels(submodels, fontsize=12)

# 実質金利(FAIL)を赤字に
labels = ax.get_yticklabels()
labels[0].set_color(C_RED)
labels[0].set_fontweight('bold')

# Status + attempt annotations
for i, (a, s) in enumerate(zip(attempts_sub, status)):
    color = C_GREEN_DARK if s == 'PASS' else C_RED
    label = f'{a}回 {s}'
    ax.text(1.65, i, label, va='center', fontsize=10, color=color, fontweight='bold')

# DA value labels
for i, d in enumerate(da_delta):
    if d >= 0:
        ax.text(d + 0.02, i + bar_height/2, f'+{d:.2f}', va='center', fontsize=9, color=C_BLUE, fontweight='bold')
    else:
        ax.text(d - 0.02, i + bar_height/2, f'{d:.2f}', va='center', fontsize=9, color=C_RED, fontweight='bold', ha='right')

# Sharpe value labels
for i, sh in enumerate(sharpe_delta):
    if sh >= 0:
        ax.text(sh + 0.02, i - bar_height/2, f'+{sh:.3f}', va='center', fontsize=8, color=C_ORANGE)
    else:
        ax.text(sh - 0.02, i - bar_height/2, f'{sh:.3f}', va='center', fontsize=8, color='#c0392b', ha='right')

ax.set_xlabel('改善量（正 = サブモデルが貢献）', fontsize=13, color='#333333')
ax.set_title('サブモデル Gate 3 アブレーション結果', fontsize=18, fontweight='bold', color=C_DARK, pad=15)
ax.set_xlim(-0.7, 1.9)
ax.set_axisbelow(True)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Custom legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=C_BLUE, alpha=0.82, label='DA delta (pp)'),
    Patch(facecolor=C_ORANGE, alpha=0.82, label='Sharpe delta'),
]
ax.legend(handles=legend_elements, fontsize=11, loc='lower right',
          framealpha=0.9, edgecolor='#CCCCCC')

# Highlight background for FAIL row
ax.axhspan(-0.5, 0.5, color='#FFE8E8', alpha=0.4, zorder=0)

# Note
ax.text(-0.65, -0.9, '※ 正の値 = そのサブモデルを除外するとメタモデル性能が劣化する = 貢献している',
        fontsize=9, color='#888888', transform=ax.transData)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'submodel_gate3_results.png'), dpi=180, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Chart 2 saved: submodel_gate3_results.png")


# ============================================================
# Chart 3: メタモデル試行ごとの性能推移 - draw.io 4パネル
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.patch.set_facecolor('white')
fig.suptitle('メタモデル性能推移（8回のイテレーション）', fontsize=20, fontweight='bold', color=C_DARK, y=0.98)

attempts = [1, 2, 3, 4, 5, 7, 8, 9]
da_values = [54.10, 57.26, 53.30, 55.35, 56.77, 60.04, 58.73, 58.73]
hcda_values = [54.30, 55.26, 59.21, 42.86, 57.61, 64.13, 61.96, 64.13]
sharpe_values = [0.428, 1.584, 1.220, 1.628, 1.834, 2.464, 2.06, 2.06]
mae_values = [0.978, 0.688, 0.717, 0.687, 0.952, 0.943, 0.942, 0.943]

x_labels = [f'#{a}' for a in attempts]
x = np.arange(len(attempts))
best_idx = 5  # Attempt #7


def draw_panel(ax, values, target, title, ylabel, target_label, baseline=None, baseline_label=None,
               pass_check=None, invert=False, fmt='.1f'):
    """共通パネル描画関数"""
    if pass_check is None:
        pass_check = lambda v: v >= target

    colors = []
    for i, v in enumerate(values):
        if i == best_idx:
            colors.append(C_GREEN_DARK)
        elif pass_check(v):
            colors.append(C_GREEN)
        else:
            colors.append(C_RED)

    bars = ax.bar(x, values, color=colors, alpha=0.85, edgecolor='white', linewidth=0.8, width=0.7, zorder=3)
    # Best bar highlight
    bars[best_idx].set_edgecolor(C_GREEN_BORDER)
    bars[best_idx].set_linewidth(2.5)
    bars[best_idx].set_alpha(1.0)

    ax.axhline(y=target, color=C_BLUE, linestyle='--', linewidth=2, alpha=0.8, label=target_label, zorder=2)
    if baseline is not None:
        ax.axhline(y=baseline, color=C_GRAY, linestyle=':', linewidth=1.5, alpha=0.7, label=baseline_label, zorder=2)

    ax.set_ylabel(ylabel, fontsize=12, color='#555555')
    ax.set_title(title, fontsize=14, fontweight='bold', color=C_DARK, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9, edgecolor='#DDDDDD')
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=0.2, linestyle='--')

    # Value labels
    for i, v in enumerate(values):
        label = f'{v:{fmt}}'
        if i == best_idx:
            label += ' *'
        color_text = colors[i]
        offset = 0.3 if not invert else -0.3
        ax.text(i, v + offset, label, ha='center', fontsize=9, fontweight='bold', color=color_text)

    # Highlight best attempt label
    tick_labels = ax.get_xticklabels()
    tick_labels[best_idx].set_color(C_GREEN_DARK)
    tick_labels[best_idx].set_fontweight('bold')


# DA panel
draw_panel(axes[0, 0], da_values, 56.0, 'Direction Accuracy', '方向精度 (%)',
           '目標 56%', baseline=43.54, baseline_label='ベースライン 43.5%')
axes[0, 0].set_ylim(40, 66)

# HCDA panel
draw_panel(axes[0, 1], hcda_values, 60.0, 'High-Confidence DA', '高確信度方向精度 (%)',
           '目標 60%')
axes[0, 1].set_ylim(35, 70)

# Sharpe panel
draw_panel(axes[1, 0], sharpe_values, 0.80, 'Sharpe Ratio（取引コスト控除後）', 'Sharpe Ratio',
           '目標 0.80', baseline=-1.70, baseline_label='ベースライン -1.70', fmt='.2f')
axes[1, 0].set_ylim(-0.2, 3.0)

# MAE panel
draw_panel(axes[1, 1], mae_values, 0.75, 'Mean Absolute Error', 'MAE (%)',
           '目標 < 0.75%', pass_check=lambda v: v < 0.75, fmt='.3f')
axes[1, 1].set_ylim(0.5, 1.15)
# MAE note
axes[1, 1].annotate('※テストセット拡張で\n目標達成が構造的に困難',
                     xy=(0.97, 0.95), xycoords='axes fraction',
                     fontsize=9, color=C_GRAY_DARK, ha='right', va='top',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF9C4', alpha=0.9, edgecolor='#d6b656', linewidth=1))

plt.tight_layout(rect=[0, 0.04, 1, 0.95])
fig.text(0.5, 0.015,
         '■ 緑 = 目標達成    ■ 赤 = 目標未達    ■ 濃い緑枠 = 最終採用モデル（Attempt #7）',
         ha='center', fontsize=11, color='#555555')

plt.savefig(os.path.join(output_dir, 'meta_model_performance.png'), dpi=180, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Chart 3 saved: meta_model_performance.png")


# ============================================================
# Chart 4: ベースライン vs 最終モデル - draw.ioスタイル
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 7))
fig.patch.set_facecolor('white')
fig.suptitle('ベースライン vs 最終モデル', fontsize=22, fontweight='bold', color=C_DARK, y=0.97)
fig.text(0.5, 0.92, 'XGBoost（9特徴量のみ） vs XGBoost + 11サブモデル出力（24特徴量）',
         ha='center', fontsize=12, color='#888888')

metrics = ['方向精度\n(DA)', '高確信度DA\n(HCDA)', 'Sharpe\nRatio']
baseline_vals = [43.54, 42.74, -1.70]
final_vals = [60.04, 64.13, 2.46]
improvements = ['+16.50pp', '+21.39pp', '+4.16']
units = ['%', '%', '']

for idx, ax in enumerate(axes):
    ax.set_facecolor('#FAFAFA')

    b_val = baseline_vals[idx]
    f_val = final_vals[idx]

    # Bars
    bars = ax.bar([0, 1], [b_val, f_val], width=0.55,
                   color=[C_GRAY_LIGHT, C_GREEN_DARK],
                   edgecolor=[C_GRAY, C_GREEN_BORDER],
                   linewidth=2, alpha=0.9, zorder=3)

    # Value labels
    for bar_idx, (bar, val) in enumerate(zip(bars, [b_val, f_val])):
        y_pos = val + 0.8 if val >= 0 else val - 2.5
        color = C_GRAY_DARK if bar_idx == 0 else C_GREEN_DARK
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{val:.2f}{units[idx]}', ha='center', va='bottom',
                fontsize=14, fontweight='bold', color=color, zorder=4)

    # Improvement arrow & label
    ax.annotate(improvements[idx],
                xy=(1, f_val), xytext=(0.5, f_val + 6),
                fontsize=16, fontweight='bold', color=C_GREEN_DARK, ha='center',
                arrowprops=dict(arrowstyle='->', color=C_GREEN_DARK, lw=2.5),
                zorder=4)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['ベースライン', '最終モデル'], fontsize=11)
    ax.set_title(metrics[idx].replace('\n', ' '), fontsize=15, fontweight='bold', color=C_DARK, pad=12)
    ax.axhline(y=0, color='#CCCCCC', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Set y limits
axes[0].set_ylim(30, 75)
axes[1].set_ylim(30, 80)
axes[2].set_ylim(-4, 10)

# Bottom summary box
fig.text(0.5, 0.02,
         'サブモデルによる文脈情報の追加で、全指標が大幅に改善',
         ha='center', fontsize=13, fontweight='bold', color=C_GREEN_DARK,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9', alpha=0.9, edgecolor=C_GREEN_DARK, linewidth=1.5))

plt.tight_layout(rect=[0, 0.06, 1, 0.89])
plt.savefig(os.path.join(output_dir, 'baseline_vs_final.png'), dpi=180, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Chart 4 saved: baseline_vs_final.png")


# ============================================================
# Chart 5: 過学習の改善推移 - draw.ioスタイル
# ============================================================
fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor('white')

attempts_overfit = [1, 2, 3, 5, 7]
train_da = [94.3, 62.79, 79.26, 64.12, 54.76]
test_da = [54.10, 57.26, 53.30, 56.77, 60.04]
gap = [t - te for t, te in zip(train_da, test_da)]

x = np.arange(len(attempts_overfit))
width = 0.32

# Background highlight for bad gaps
for i, g in enumerate(gap):
    if g > 10:
        ax.axvspan(i - 0.48, i + 0.48, color='#FFEBEE', alpha=0.5, zorder=0)

# Bars
bars_train = ax.bar(x - width/2, train_da, width, label='Train DA',
                     color=C_RED, alpha=0.72, edgecolor='white', linewidth=0.8, zorder=3)
bars_test = ax.bar(x + width/2, test_da, width, label='Test DA',
                    color=C_BLUE, alpha=0.72, edgecolor='white', linewidth=0.8, zorder=3)

# Highlight Attempt #7 bars
bars_train[4].set_alpha(0.9)
bars_test[4].set_edgecolor(C_BLUE_DARK)
bars_test[4].set_linewidth(2.5)
bars_test[4].set_alpha(0.95)

# Gap annotations (drawn as boxes)
for i, g in enumerate(gap):
    color = C_RED if g > 10 else C_GREEN_DARK
    y_pos = max(train_da[i], test_da[i]) + 2
    gap_text = f'gap: {g:.1f}pp'
    ax.text(i, y_pos, gap_text, ha='center', fontsize=11, fontweight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor=color, linewidth=1.2))

# Target line
ax.axhline(y=56.0, color=C_GREEN, linestyle='--', linewidth=2, alpha=0.7, label='DA目標 56%', zorder=2)

ax.set_ylabel('方向精度 (%)', fontsize=13, color='#555555')
ax.set_title('過学習の改善推移（Train-Test DA Gap）', fontsize=18, fontweight='bold', color=C_DARK, pad=15)
ax.set_xticks(x)
ax.set_xticklabels([f'#{a}' for a in attempts_overfit], fontsize=12)
ax.set_xlabel('Attempt', fontsize=13, color='#555555')
ax.set_ylim(42, 100)
ax.legend(fontsize=11, loc='upper right', framealpha=0.9, edgecolor='#DDDDDD')
ax.set_axisbelow(True)
ax.grid(axis='y', alpha=0.2, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Highlight Attempt #7 tick
tick_labels = ax.get_xticklabels()
tick_labels[4].set_color(C_GREEN_DARK)
tick_labels[4].set_fontweight('bold')
tick_labels[4].set_fontsize(14)

# Attempt #7 callout
highlight_box = FancyBboxPatch((3.5, 82), 2.2, 12, boxstyle="round,pad=0.3",
                                facecolor='#E8F5E9', alpha=0.9, edgecolor=C_GREEN_DARK,
                                linewidth=1.5, linestyle='--', zorder=4)
ax.add_patch(highlight_box)
ax.text(4.6, 90.5, 'Attempt #7', ha='center', fontsize=12, fontweight='bold', color=C_GREEN_DARK, zorder=5)
ax.text(4.6, 85, 'TestがTrainを上回る！', ha='center', fontsize=11, color=C_GREEN_DARK, zorder=5)
ax.annotate('', xy=(4 + width/2, 61), xytext=(4.6, 82),
            arrowprops=dict(arrowstyle='->', color=C_GREEN_DARK, lw=2.5), zorder=5)

# Bottom legend note
fig.text(0.5, 0.01,
         '赤背景 = 過学習が深刻（gap > 10pp）  |  gap = Train DA - Test DA',
         ha='center', fontsize=10, color='#888888')

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(os.path.join(output_dir, 'overfitting_progress.png'), dpi=180, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Chart 5 saved: overfitting_progress.png")


print("\n=== All 5 charts generated successfully (draw.io style v2) ===")
