"""
Create evaluation entry after Gate evaluation
Called by evaluator agent
"""

import sys
import json
from pathlib import Path
from datetime import datetime

def create_evaluation_entry(feature, attempt, gate_results, analysis):
    """
    Create evaluation entry in knowledge base

    Args:
        feature: str, feature name (e.g., 'real_rate')
        attempt: int, attempt number
        gate_results: dict, results from Gate 1/2/3
        analysis: dict, analysis from evaluator
    """

    # Read training results
    training_log = Path(f"logs/training/{feature}_{attempt}.json")
    if training_log.exists():
        with open(training_log, 'r') as f:
            training_data = json.load(f)
    else:
        training_data = {}

    # Determine overall result
    gates_passed = []
    gates_failed = []
    for gate_num in [1, 2, 3]:
        if f'gate{gate_num}' in gate_results:
            if gate_results[f'gate{gate_num}']['status'] == 'PASS':
                gates_passed.append(gate_num)
            else:
                gates_failed.append(gate_num)

    # Create evaluation content
    content = f"""## Evaluation: {feature} Attempt {attempt} - {datetime.now().strftime('%Y-%m-%d')}

**Context**: {feature} submodel, Attempt {attempt}, Phase {analysis.get('phase', 'Unknown')}

**Approach**:
- Architecture: {training_data.get('architecture', 'Unknown')}
- Input shape: {training_data.get('input_shape', 'Unknown')}
- Latent dimensions: {training_data.get('latent_dim', 'Unknown')}
- Best hyperparameters:
{format_hyperparameters(training_data.get('best_params', {}))}

**Result**:
{format_gate_results(gate_results, gates_passed, gates_failed)}

**Analysis**:

What worked:
{format_list(analysis.get('what_worked', []))}

What didn't work:
{format_list(analysis.get('what_didnt_work', []))}

Why (hypothesis):
{format_list(analysis.get('hypothesis', []))}

**Recommendation**:
{format_recommendations(analysis.get('recommendations', {}))}

**Evidence**:
- Training log: logs/training/{feature}_{attempt}.json
- Evaluation log: logs/evaluation/{feature}_{attempt}.json
- Submodel output: data/submodel_outputs/{feature}.csv

**Next Steps**:
{format_list(analysis.get('next_steps', []))}

**Last Updated**: {datetime.now().strftime('%Y-%m-%d')}

---

"""

    # Append to submodel evaluation file
    eval_file = Path(f"docs/knowledge/evaluations/submodels/{feature}.md")

    if not eval_file.exists():
        # Create header
        header = f"# {feature} Submodel Evaluations\n\n"
        header += "Chronological record of all attempts and learnings.\n\n"
        header += "---\n\n"
        content = header + content

    with open(eval_file, 'a', encoding='utf-8') as f:
        f.write(content)

    print(f"Created evaluation entry: {eval_file}")
    return eval_file

def format_hyperparameters(params):
    """Format hyperparameters as bullet list"""
    if not params:
        return "  - None recorded"

    lines = []
    for key, value in params.items():
        lines.append(f"  - {key}: {value}")
    return '\n'.join(lines)

def format_gate_results(results, passed, failed):
    """Format gate results"""
    lines = []
    for gate_num in [1, 2, 3]:
        if f'gate{gate_num}' in results:
            status = results[f'gate{gate_num}']['status']
            details = results[f'gate{gate_num}'].get('details', {})

            if status == 'PASS':
                lines.append(f"- Gate {gate_num}: ✓ PASS")
            else:
                lines.append(f"- Gate {gate_num}: ✗ FAIL")

            # Add key metrics
            for key, value in details.items():
                if isinstance(value, float):
                    lines.append(f"  - {key}: {value:.4f}")
                else:
                    lines.append(f"  - {key}: {value}")

    if not lines:
        return "- No gate results available"

    return '\n'.join(lines)

def format_list(items):
    """Format list items"""
    if not items:
        return "- (None identified)"

    return '\n'.join([f"- {item}" for item in items])

def format_recommendations(recs):
    """Format recommendations"""
    lines = []

    if 'use_for' in recs and recs['use_for']:
        lines.append(f"✓ Use for: {', '.join(recs['use_for'])}")

    if 'avoid_for' in recs and recs['avoid_for']:
        lines.append(f"✗ Avoid for: {', '.join(recs['avoid_for'])}")

    if 'alternative' in recs and recs['alternative']:
        lines.append(f"Alternative: {recs['alternative']}")

    if not lines:
        return "- (No specific recommendations)"

    return '\n'.join(lines)

if __name__ == "__main__":
    # Example usage
    if len(sys.argv) < 3:
        print("Usage: python scripts/create_evaluation_entry.py <feature> <attempt>")
        print("\nExample:")
        print("  python scripts/create_evaluation_entry.py real_rate 1")
        sys.exit(1)

    feature = sys.argv[1]
    attempt = int(sys.argv[2])

    # Dummy data for testing
    gate_results = {
        'gate1': {
            'status': 'PASS',
            'details': {'overfit_ratio': 1.2, 'leakage': False}
        },
        'gate2': {
            'status': 'FAIL',
            'details': {'mi_increase': 0.023, 'target': 0.05}
        }
    }

    analysis = {
        'phase': 'smoke_test',
        'what_worked': ['GPU training fast', 'No overfitting'],
        'what_didnt_work': ['Low information gain', 'High correlation with input'],
        'hypothesis': ['Window size too small', 'Trivial identity mapping'],
        'recommendations': {
            'use_for': ['Quick smoke tests'],
            'avoid_for': ['Complex temporal patterns'],
            'alternative': 'Try GRU with attention'
        },
        'next_steps': ['Increase window to 120 days', 'Add contrastive loss']
    }

    create_evaluation_entry(feature, attempt, gate_results, analysis)
