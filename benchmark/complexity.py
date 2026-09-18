from __future__ import annotations

import re
from typing import Literal


def count_operators(equation: str) -> int:
    """
    Count the number of binary operators in an equation string.

    This is a simple but practical complexity metric:
    - Counts +, -, *, / operators
    - Returns count + 1 (for the intercept/constant term)

    Example:
        "x0 + x1*u0 - x0**2 / u1" → 4 operators + 1 = 5
    """
    eq_str = str(equation).strip()
    operator_count = 0
    for op in ["+", "-", "*", "/"]:
        operator_count += eq_str.count(op)
    return operator_count + 1


def calculate_complexity(
    equation: str,
    method: Literal["operator_count", "symantic_inspired"] = "operator_count",
) -> float:
    """
    Calculate equation complexity using specified method.

    Args:
        equation: Equation string (e.g., from SINDy or SyMANTIC)
        method: Complexity metric to use:
            - "operator_count": Simple count of +, -, *, / operators + 1
            - "symantic_inspired": Information-theoretic approximation inspired by SyMANTIC
                (based on operator count and diversity)

    Returns:
        Complexity value (float)

    Notes:
        SyMANTIC uses: complexity = (num_operators + num_unique_operators) * log2(unique_count + 1)
        For base features: complexity = 1
        This function provides a simplified version for equation comparison.
    """
    eq_str = str(equation).strip()

    if method == "operator_count":
        return float(count_operators(eq_str))

    elif method == "symantic_inspired":
        # Extract all operators in order
        operators = re.findall(r"[\+\-\*/]", eq_str)
        if not operators:
            return 1.0

        num_operators = len(operators)
        unique_operators = len(set(operators))
        import math

        # Approximate SyMANTIC's formula for equation complexity
        complexity = (num_operators + unique_operators) * math.log2(unique_operators + 1)
        return float(complexity)

    else:
        raise ValueError(f"Unknown complexity method: {method}")


def complexity_comparison(
    equation_1: str,
    equation_2: str,
    method: Literal["operator_count", "symantic_inspired"] = "operator_count",
) -> dict[str, float | tuple[float, float]]:
    """
    Compare complexity of two equations side-by-side.

    Args:
        equation_1, equation_2: Two equations to compare
        method: Complexity calculation method

    Returns:
        Dictionary with:
            - "eq1": equation string
            - "eq2": equation string
            - "complexity_1": complexity of equation 1
            - "complexity_2": complexity of equation 2
            - "total_complexity": sum of both complexities
            - "complexity_ratio": eq1 complexity / eq2 complexity
    """
    c1 = calculate_complexity(equation_1, method=method)
    c2 = calculate_complexity(equation_2, method=method)

    ratio = c1 / c2 if c2 != 0 else float("inf")

    return {
        "eq1": str(equation_1),
        "eq2": str(equation_2),
        "complexity_1": c1,
        "complexity_2": c2,
        "total_complexity": c1 + c2,
        "complexity_ratio": ratio,
    }
