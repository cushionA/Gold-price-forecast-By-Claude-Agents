#!/usr/bin/env python3
"""
Notebook Validation Script for Kaggle Submission
Detects common errors before expensive Kaggle execution

Usage:
    python scripts/validate_notebook.py notebooks/meta_model_6/
"""

import json
import ast
import re
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any


class NotebookValidator:
    """Validates Jupyter notebooks for Kaggle submission"""

    # Known problematic patterns (only detect ALL-CAPS method names, not valid .upper() usage)
    TYPO_PATTERNS = [
        (r'\w+\.UPPER\(\)', '.upper()', 'Method should be lowercase: .upper() - found variable.UPPER() instead'),
        (r'\w+\.LOWER\(\)', '.lower()', 'Method should be lowercase: .lower() - found variable.LOWER() instead'),
        (r'\w+\.SPLIT\(\)', '.split()', 'Method should be lowercase: .split() - found variable.SPLIT() instead'),
        (r'\w+\.STRIP\(\)', '.strip()', 'Method should be lowercase: .strip() - found variable.STRIP() instead'),
        (r'\w+\.REPLACE\(', '.replace(', 'Method should be lowercase: .replace() - found variable.REPLACE( instead'),
    ]

    # Known compatibility issues
    COMPATIBILITY_WARNINGS = [
        {
            'pattern': r'import shap|from shap',
            'condition': r'xgboost',
            'message': 'SHAP + XGBoost 2.x compatibility issue detected. Consider using bootstrap ensemble instead.',
        },
    ]

    def __init__(self, notebook_dir: str):
        self.notebook_dir = Path(notebook_dir)
        self.notebook_path = self.notebook_dir / 'train.ipynb'
        self.metadata_path = self.notebook_dir / 'kernel-metadata.json'

        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self) -> Tuple[List[str], List[str]]:
        """Run all validation checks"""
        print(f"[VALIDATE] Validating notebook: {self.notebook_path}")
        print("=" * 60)

        # 1. File existence
        self._check_file_existence()

        if not self.errors:
            # Load notebook
            try:
                with open(self.notebook_path, 'r', encoding='utf-8') as f:
                    self.notebook = json.load(f)
            except Exception as e:
                self.errors.append(f"Failed to load notebook: {e}")
                return self.errors, self.warnings

            # 2. Syntax check
            self._check_python_syntax()

            # 3. Typo detection
            self._check_common_typos()

            # 4. Compatibility warnings
            self._check_compatibility()

            # 5. Dataset references
            self._check_dataset_references()

            # 6. kernel-metadata.json validation
            self._check_kernel_metadata()

            # 7. Undefined variables (basic)
            self._check_undefined_variables()

        return self.errors, self.warnings

    def _check_file_existence(self):
        """Check if required files exist"""
        if not self.notebook_path.exists():
            self.errors.append(f"Notebook not found: {self.notebook_path}")

        if not self.metadata_path.exists():
            self.errors.append(f"kernel-metadata.json not found: {self.metadata_path}")

    def _check_python_syntax(self):
        """Check Python syntax in code cells"""
        print("\n1. Checking Python syntax...")

        for cell_idx, cell in enumerate(self.notebook.get('cells', [])):
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))

                # Skip cells with shell commands or magic commands
                if source.strip().startswith(('!', '%', '%%')):
                    continue

                try:
                    ast.parse(source)
                except SyntaxError as e:
                    self.errors.append(
                        f"Syntax error in cell {cell_idx}: {e.msg} at line {e.lineno}"
                    )

        if not any('Syntax error' in e for e in self.errors):
            print("   [OK] No syntax errors found")

    def _check_common_typos(self):
        """Detect common typo patterns"""
        print("\n2. Checking for common typos...")

        found_typos = []

        for cell_idx, cell in enumerate(self.notebook.get('cells', [])):
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))

                for pattern, correct, message in self.TYPO_PATTERNS:
                    matches = re.finditer(pattern, source)
                    for match in matches:
                        line_num = source[:match.start()].count('\n') + 1
                        found_typos.append(
                            f"Cell {cell_idx}, line {line_num}: {message}"
                        )

        if found_typos:
            self.warnings.extend(found_typos)
            print(f"   [WARN]  Found {len(found_typos)} potential typos")
        else:
            print("   [OK] No common typos detected")

    def _check_compatibility(self):
        """Check for known compatibility issues"""
        print("\n3. Checking library compatibility...")

        full_source = self._get_full_source()

        for compat in self.COMPATIBILITY_WARNINGS:
            if re.search(compat['pattern'], full_source):
                if re.search(compat['condition'], full_source):
                    self.warnings.append(compat['message'])

        if any('compatibility' in w.lower() for w in self.warnings):
            print(f"   [WARN]  Found compatibility warnings")
        else:
            print("   [OK] No known compatibility issues")

    def _check_dataset_references(self):
        """Check dataset path references"""
        print("\n4. Checking dataset references...")

        full_source = self._get_full_source()

        # Find all dataset path references
        dataset_patterns = [
            r'\.\.\/input\/([a-z0-9-]+)\/',
            r'\.\.\/input\/([a-z0-9-]+)\.csv',
        ]

        dataset_refs = set()
        for pattern in dataset_patterns:
            matches = re.finditer(pattern, full_source)
            for match in matches:
                dataset_refs.add(match.group(1))

        if dataset_refs:
            print(f"   [INFO] Found {len(dataset_refs)} dataset reference(s): {', '.join(dataset_refs)}")

            # Check if referenced in metadata
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)

                dataset_sources = metadata.get('dataset_sources', [])
                for ref in dataset_refs:
                    # Check if any dataset source contains this reference
                    if not any(ref in ds for ds in dataset_sources):
                        self.warnings.append(
                            f"Dataset '{ref}' referenced in notebook but not in kernel-metadata.json dataset_sources"
                        )
        else:
            print("   [OK] No dataset references found (using API fetching)")

    def _check_kernel_metadata(self):
        """Validate kernel-metadata.json"""
        print("\n5. Validating kernel-metadata.json...")

        if not self.metadata_path.exists():
            return

        try:
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            self.errors.append(f"Failed to load kernel-metadata.json: {e}")
            return

        # Required fields
        required_fields = ['id', 'title', 'code_file', 'language', 'kernel_type']
        for field in required_fields:
            if field not in metadata:
                self.errors.append(f"kernel-metadata.json missing required field: {field}")

        # Check code_file matches
        if metadata.get('code_file') != 'train.ipynb':
            self.warnings.append(
                f"code_file is '{metadata.get('code_file')}' but notebook is 'train.ipynb'"
            )

        # Check language
        if metadata.get('language') != 'python':
            self.warnings.append(f"language is '{metadata.get('language')}', expected 'python'")

        # Check kernel_type
        if metadata.get('kernel_type') != 'notebook':
            self.warnings.append(f"kernel_type is '{metadata.get('kernel_type')}', expected 'notebook'")

        # Check GPU setting
        if metadata.get('enable_gpu') and 'cuda' not in self._get_full_source().lower():
            self.warnings.append("GPU enabled but no CUDA/GPU code detected")

        if not self.errors and not any('kernel-metadata' in w for w in self.warnings):
            print("   [OK] kernel-metadata.json is valid")
        elif self.warnings:
            print("   [WARN]  kernel-metadata.json has warnings")

    def _check_undefined_variables(self):
        """Basic check for potentially undefined variables"""
        print("\n6. Checking for undefined variables (basic)...")

        # This is a simplified check - only catches obvious cases
        defined_vars = set()
        used_vars = set()

        for cell in self.notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))

                # Skip magic/shell commands
                if source.strip().startswith(('!', '%', '%%')):
                    continue

                try:
                    tree = ast.parse(source)

                    # Find assignments
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Assign):
                            for target in node.targets:
                                if isinstance(target, ast.Name):
                                    defined_vars.add(target.id)
                        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                            # Variable is being used
                            used_vars.add(node.id)

                except SyntaxError:
                    pass  # Already caught in syntax check

        # Builtins and common imports
        builtins = {
            'print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict', 'set',
            'np', 'pd', 'plt', 'os', 'sys', 'json', 'datetime', 'Path',
            'True', 'False', 'None', '__name__', '__file__',
        }

        undefined = used_vars - defined_vars - builtins

        # Filter out common false positives
        undefined = {v for v in undefined if not v.startswith('_') and len(v) > 1}

        if undefined and len(undefined) < 20:  # Only report if reasonable number
            self.warnings.append(
                f"Potentially undefined variables: {', '.join(sorted(list(undefined)[:10]))}"
            )

        if not any('undefined' in w.lower() for w in self.warnings):
            print("   [OK] No obvious undefined variables")

    def _get_full_source(self) -> str:
        """Get concatenated source code from all cells"""
        source_parts = []
        for cell in self.notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                source_parts.append(''.join(cell.get('source', [])))
        return '\n'.join(source_parts)

    def print_report(self):
        """Print validation report"""
        print("\n" + "=" * 60)
        print("VALIDATION REPORT")
        print("=" * 60)

        if self.errors:
            print(f"\n[FAIL] CRITICAL ERRORS ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                print(f"   {i}. {error}")
            print("\n[WARN]  Fix these before Kaggle submission!")
        else:
            print("\n[PASS] No critical errors found")

        if self.warnings:
            print(f"\n[WARN]  WARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"   {i}. {warning}")
            print("\n[TIP] Review these warnings (non-blocking)")
        else:
            print("\n[PASS] No warnings")

        print("\n" + "=" * 60)

        if not self.errors:
            print("[PASS] Notebook is ready for Kaggle submission")
        else:
            print("[FAIL] Notebook has errors - DO NOT submit to Kaggle")

        print("=" * 60)

        return len(self.errors) == 0


def validate_notebook(notebook_dir: str) -> Tuple[bool, List[str], List[str]]:
    """
    Validate a Jupyter notebook for Kaggle submission

    Args:
        notebook_dir: Directory containing train.ipynb and kernel-metadata.json

    Returns:
        (is_valid, errors, warnings)
    """
    validator = NotebookValidator(notebook_dir)
    errors, warnings = validator.validate()
    is_valid = validator.print_report()

    return is_valid, errors, warnings


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_notebook.py <notebook_directory>")
        print("Example: python scripts/validate_notebook.py notebooks/meta_model_6/")
        sys.exit(1)

    notebook_dir = sys.argv[1]
    is_valid, errors, warnings = validate_notebook(notebook_dir)

    # Exit with error code if validation failed
    sys.exit(0 if is_valid else 1)
