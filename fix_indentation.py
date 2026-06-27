#!/usr/bin/env python3
"""Fix indentation issues in existing ablation results.

The original results have misaligned control structures (elif/else at wrong indent).
This script re-processes the generated_code field to fix these issues.
"""

import json
import re
import ast
import textwrap
from pathlib import Path


def fix_indentation(code: str) -> str:
    """Fix control structure alignment in generated code.

    Handles cases where elif/else/except are indented incorrectly.
    """
    if not code or not code.strip():
        return "    pass"

    # Dedent completely first
    dedented = textwrap.dedent(code)
    lines = dedented.split('\n')

    # Fix control structure alignment
    control_keywords = {'elif', 'else', 'except', 'finally'}
    block_starters = {'if', 'try', 'for', 'while', 'with', 'match'}

    fixed_lines = []
    indent_stack = []  # Track (keyword, indent_level) for block matching

    for line in lines:
        stripped = line.strip()
        if not stripped:
            fixed_lines.append('')
            continue

        current_indent = len(line) - len(line.lstrip())
        first_word = stripped.split()[0].rstrip(':') if stripped.split() else ''

        # Check if this is a continuation control keyword
        if first_word in control_keywords:
            # Find the matching opener's indentation
            while indent_stack:
                opener, opener_indent = indent_stack[-1]
                if opener_indent <= current_indent:
                    valid_pairs = {
                        'elif': {'if', 'elif'},
                        'else': {'if', 'elif', 'for', 'while', 'try', 'except'},
                        'except': {'try', 'except'},
                        'finally': {'try', 'except', 'else'},
                    }
                    if opener in valid_pairs.get(first_word, set()):
                        current_indent = opener_indent
                        break
                indent_stack.pop()

        # Track block starters
        if first_word in block_starters or first_word in {'elif', 'except'}:
            indent_stack.append((first_word, current_indent))
        elif first_word == 'else' and ':' in stripped:
            indent_stack.append(('else', current_indent))

        # Reconstruct with 4-space base
        fixed_lines.append(' ' * (4 + current_indent) + stripped)

    result = '\n'.join(fixed_lines)

    # Validate
    try:
        test_code = "def _test_func():\n" + result
        ast.parse(test_code)
        return result
    except SyntaxError:
        # Fallback: just ensure 4-space base indent
        return '\n'.join('    ' + line.strip() if line.strip() else '' for line in lines)


def process_file(input_path: Path, output_path: Path):
    """Process a JSONL file and fix all generated_code fields."""
    fixed = 0
    unchanged = 0
    errors = 0

    results = []
    with open(input_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                original_code = record.get('generated_code', '')
                fixed_code = fix_indentation(original_code)

                if fixed_code != original_code:
                    fixed += 1
                    record['generated_code'] = fixed_code
                    record['indentation_fixed'] = True
                else:
                    unchanged += 1
                    record['indentation_fixed'] = False

                # Validate the fixed code
                try:
                    test_code = "def _test_func():\n" + fixed_code
                    ast.parse(test_code)
                    record['syntax_valid'] = True
                except SyntaxError:
                    record['syntax_valid'] = False
                    errors += 1

                results.append(record)
            except json.JSONDecodeError:
                continue

    with open(output_path, 'w') as f:
        for record in results:
            f.write(json.dumps(record) + '\n')

    return fixed, unchanged, errors


def main():
    input_dir = Path('results/ablation')
    output_dir = Path('results/ablation_fixed')
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("FIXING INDENTATION IN ABLATION RESULTS")
    print("=" * 60)

    for input_file in input_dir.glob('*.jsonl'):
        output_file = output_dir / input_file.name
        fixed, unchanged, errors = process_file(input_file, output_file)
        print(f"\n{input_file.name}:")
        print(f"  Fixed:     {fixed}")
        print(f"  Unchanged: {unchanged}")
        print(f"  Still invalid: {errors}")

    print("\n" + "=" * 60)
    print(f"Fixed results saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
