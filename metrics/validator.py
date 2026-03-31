"""
Code Validation Module for BddAgent

Provides syntax validation, code quality metrics, and semantic analysis
for generated Python code.
"""

import ast
import re
import sys
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from io import StringIO


class ValidationLevel(Enum):
    """Validation strictness levels."""
    SYNTAX_ONLY = "syntax_only"
    BASIC = "basic"  # Syntax + structure
    STANDARD = "standard"  # + code quality metrics
    STRICT = "strict"  # + style checks


@dataclass
class ValidationResult:
    """Detailed validation results."""
    # Core validation
    syntax_valid: bool = False
    syntax_error: Optional[str] = None
    syntax_error_line: Optional[int] = None

    # Structure checks
    has_function_body: bool = False
    has_return_statement: bool = False
    has_proper_indentation: bool = False
    uses_pass_only: bool = False

    # Code metrics
    lines_of_code: int = 0
    blank_lines: int = 0
    comment_lines: int = 0
    cyclomatic_complexity: int = 1
    nesting_depth: int = 0

    # Quality indicators
    has_docstring: bool = False
    has_type_hints: bool = False
    uses_exception_handling: bool = False

    # Warnings and issues
    warnings: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if code passes validation."""
        return self.syntax_valid and self.has_function_body and not self.uses_pass_only

    @property
    def quality_score(self) -> float:
        """Compute overall quality score (0-1)."""
        if not self.syntax_valid:
            return 0.0

        score = 0.5  # Base score for valid syntax

        # Positive factors
        if self.has_return_statement:
            score += 0.1
        if self.has_proper_indentation:
            score += 0.1
        if not self.uses_pass_only:
            score += 0.1
        if self.has_docstring:
            score += 0.05
        if self.has_type_hints:
            score += 0.05
        if self.uses_exception_handling:
            score += 0.05

        # Negative factors
        if self.nesting_depth > 4:
            score -= 0.05 * (self.nesting_depth - 4)
        if self.cyclomatic_complexity > 10:
            score -= 0.05 * (self.cyclomatic_complexity - 10)

        # Penalize for warnings
        score -= 0.02 * len(self.warnings)

        return max(0.0, min(1.0, score))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "syntax_valid": self.syntax_valid,
            "syntax_error": self.syntax_error,
            "syntax_error_line": self.syntax_error_line,
            "has_function_body": self.has_function_body,
            "has_return_statement": self.has_return_statement,
            "has_proper_indentation": self.has_proper_indentation,
            "uses_pass_only": self.uses_pass_only,
            "lines_of_code": self.lines_of_code,
            "blank_lines": self.blank_lines,
            "comment_lines": self.comment_lines,
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "nesting_depth": self.nesting_depth,
            "has_docstring": self.has_docstring,
            "has_type_hints": self.has_type_hints,
            "uses_exception_handling": self.uses_exception_handling,
            "warnings": self.warnings,
            "issues": self.issues,
            "is_valid": self.is_valid,
            "quality_score": self.quality_score
        }


class CodeValidator:
    """
    Validates generated Python code for syntax, structure, and quality.
    """

    def __init__(self, level: ValidationLevel = ValidationLevel.STANDARD):
        self.level = level

    def validate(self, code: str, wrap_as_function: bool = True) -> ValidationResult:
        """
        Validate Python code.

        Args:
            code: The code to validate (typically function body)
            wrap_as_function: If True, wraps code in a function definition for parsing

        Returns:
            ValidationResult with detailed validation information
        """
        result = ValidationResult()

        if not code or not code.strip():
            result.syntax_error = "Empty code"
            return result

        # Prepare code for parsing
        if wrap_as_function:
            full_code = f"def _validate_fn():\n{self._ensure_indentation(code)}"
        else:
            full_code = code

        # Count lines
        lines = code.split('\n')
        result.lines_of_code = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
        result.blank_lines = len([l for l in lines if not l.strip()])
        result.comment_lines = len([l for l in lines if l.strip().startswith('#')])

        # Syntax validation
        try:
            tree = ast.parse(full_code)
            result.syntax_valid = True
        except SyntaxError as e:
            result.syntax_valid = False
            result.syntax_error = str(e.msg) if hasattr(e, 'msg') else str(e)
            result.syntax_error_line = e.lineno
            return result

        # Structure checks
        result.has_function_body = self._check_has_body(code)
        result.has_proper_indentation = self._check_indentation(code)
        result.uses_pass_only = self._check_pass_only(code)

        # Skip deeper analysis for SYNTAX_ONLY level
        if self.level == ValidationLevel.SYNTAX_ONLY:
            return result

        # AST-based analysis
        try:
            if wrap_as_function:
                func_node = tree.body[0]
                if isinstance(func_node, ast.FunctionDef):
                    result.has_return_statement = self._has_return(func_node)
                    result.has_docstring = self._has_docstring(func_node)
                    result.has_type_hints = self._has_type_hints(func_node)
                    result.uses_exception_handling = self._has_try_except(func_node)

            if self.level in [ValidationLevel.STANDARD, ValidationLevel.STRICT]:
                result.cyclomatic_complexity = self._compute_complexity(tree)
                result.nesting_depth = self._compute_nesting_depth(tree)

        except Exception as e:
            result.warnings.append(f"AST analysis error: {str(e)}")

        # Quality checks for STRICT level
        if self.level == ValidationLevel.STRICT:
            self._check_style(code, result)

        return result

    def validate_function_body(
        self,
        code: str,
        requirements: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate function body specifically (wrapper for main validate method).

        Args:
            code: Function body code
            requirements: Original requirements (for semantic validation)

        Returns:
            ValidationResult
        """
        result = self.validate(code, wrap_as_function=True)

        if requirements and result.syntax_valid:
            # Basic semantic checks
            self._check_requirements_coverage(code, requirements, result)

        return result

    def _ensure_indentation(self, code: str) -> str:
        """Ensure code has proper base indentation."""
        lines = code.split('\n')
        result_lines = []

        for line in lines:
            if line.strip():  # Non-empty line
                # Check if already indented
                if not line.startswith(' ') and not line.startswith('\t'):
                    line = '    ' + line
            result_lines.append(line)

        return '\n'.join(result_lines)

    def _check_has_body(self, code: str) -> bool:
        """Check if code has actual implementation."""
        stripped = code.strip()
        if not stripped:
            return False

        # Check for actual code content
        lines = [l.strip() for l in stripped.split('\n') if l.strip()]
        non_trivial = [l for l in lines if l not in ['pass', '...'] and not l.startswith('#')]

        return len(non_trivial) > 0

    def _check_indentation(self, code: str) -> bool:
        """Check if code has consistent indentation."""
        lines = code.split('\n')

        for line in lines:
            if line.strip():  # Non-empty line
                indent = len(line) - len(line.lstrip())
                # Should be multiple of 4 or at least consistent
                if indent > 0 and indent % 4 != 0 and indent % 2 != 0:
                    return False

        return True

    def _check_pass_only(self, code: str) -> bool:
        """Check if code is just 'pass' or empty implementation."""
        stripped = code.strip()
        lines = [l.strip() for l in stripped.split('\n') if l.strip() and not l.strip().startswith('#')]

        if not lines:
            return True

        # Check if only pass statements
        return all(l in ['pass', '...', 'raise NotImplementedError', 'raise NotImplementedError()']
                   for l in lines)

    def _has_return(self, func_node: ast.FunctionDef) -> bool:
        """Check if function has a return statement."""
        for node in ast.walk(func_node):
            if isinstance(node, ast.Return) and node.value is not None:
                return True
        return False

    def _has_docstring(self, func_node: ast.FunctionDef) -> bool:
        """Check if function has a docstring."""
        if func_node.body:
            first_stmt = func_node.body[0]
            if isinstance(first_stmt, ast.Expr):
                if isinstance(first_stmt.value, ast.Constant):
                    return isinstance(first_stmt.value.value, str)
        return False

    def _has_type_hints(self, func_node: ast.FunctionDef) -> bool:
        """Check if function has type hints."""
        # Check return annotation
        if func_node.returns:
            return True

        # Check argument annotations
        for arg in func_node.args.args:
            if arg.annotation:
                return True

        return False

    def _has_try_except(self, node: ast.AST) -> bool:
        """Check if code uses try/except."""
        for child in ast.walk(node):
            if isinstance(child, (ast.Try, ast.ExceptHandler)):
                return True
        return False

    def _compute_complexity(self, tree: ast.AST) -> int:
        """Compute cyclomatic complexity."""
        complexity = 1  # Base complexity

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For,
                                 ast.ExceptHandler, ast.With,
                                 ast.Assert, ast.comprehension)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1

        return complexity

    def _compute_nesting_depth(self, tree: ast.AST, depth: int = 0) -> int:
        """Compute maximum nesting depth."""
        max_depth = depth

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                child_depth = self._compute_nesting_depth(node, depth + 1)
                max_depth = max(max_depth, child_depth)
            else:
                child_depth = self._compute_nesting_depth(node, depth)
                max_depth = max(max_depth, child_depth)

        return max_depth

    def _check_style(self, code: str, result: ValidationResult):
        """Check code style (for STRICT validation)."""
        lines = code.split('\n')

        for i, line in enumerate(lines, 1):
            # Check line length
            if len(line) > 120:
                result.warnings.append(f"Line {i}: exceeds 120 characters")

            # Check trailing whitespace
            if line.rstrip() != line:
                result.warnings.append(f"Line {i}: trailing whitespace")

            # Check tabs vs spaces
            if '\t' in line:
                result.warnings.append(f"Line {i}: uses tabs instead of spaces")

    def _check_requirements_coverage(
        self,
        code: str,
        requirements: str,
        result: ValidationResult
    ):
        """Basic check if code addresses requirements."""
        # Extract key terms from requirements
        req_lower = requirements.lower()
        code_lower = code.lower()

        # Check for common requirement keywords
        keywords = ['return', 'if', 'for', 'while', 'list', 'dict', 'string', 'int', 'float']
        mentioned_keywords = [kw for kw in keywords if kw in req_lower]

        # Simple coverage check
        for kw in mentioned_keywords:
            if kw in ['list', 'dict', 'string', 'int', 'float']:
                # Type mentions might indicate expected data structures
                if kw == 'list' and '[' not in code and 'list' not in code_lower:
                    result.warnings.append(f"Requirements mention '{kw}' but code may not use it")
                if kw == 'dict' and '{' not in code and 'dict' not in code_lower:
                    result.warnings.append(f"Requirements mention '{kw}' but code may not use it")


def validate_code(code: str, level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
    """Convenience function to validate code."""
    validator = CodeValidator(level)
    return validator.validate(code)


def validate_function_body(
    code: str,
    requirements: Optional[str] = None,
    level: ValidationLevel = ValidationLevel.STANDARD
) -> ValidationResult:
    """Convenience function to validate function body."""
    validator = CodeValidator(level)
    return validator.validate_function_body(code, requirements)
