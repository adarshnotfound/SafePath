"""Membership functions and linguistic variable representations for SafePath.

Implements triangular (trimf) and trapezoidal (trapmf) membership calculations,
fuzzification of raw numeric/categorical inputs, and linguistic variable management.
"""

from typing import Dict, List, Union, Tuple
import numpy as np
from fuzzy.config import DEFAULT_MEMBERSHIP_CONFIG


def trimf(x: Union[float, np.ndarray], params: List[float]) -> Union[float, np.ndarray]:
    """Triangular membership function.
    
    Parameters:
        x: Input value or array.
        params: 3-element list [a, b, c] where:
                a = left foot (membership = 0)
                b = peak (membership = 1)
                c = right foot (membership = 0)
    Returns:
        Membership degree in [0.0, 1.0].
    """
    a, b, c = params
    x_arr = np.asarray(x, dtype=float)
    y = np.zeros_like(x_arr)
    
    # Left slope
    if a != b:
        left_mask = (x_arr >= a) & (x_arr <= b)
        y[left_mask] = (x_arr[left_mask] - a) / (b - a)
    else:
        y[x_arr == b] = 1.0
        
    # Right slope
    if b != c:
        right_mask = (x_arr > b) & (x_arr <= c)
        y[right_mask] = (c - x_arr[right_mask]) / (c - b)
        
    # Peak point
    y[x_arr == b] = 1.0
    
    # Ensure clipping to [0, 1]
    y = np.clip(y, 0.0, 1.0)
    if np.isscalar(x):
        return float(y.item())
    return y


def trapmf(x: Union[float, np.ndarray], params: List[float]) -> Union[float, np.ndarray]:
    """Trapezoidal membership function.
    
    Parameters:
        x: Input value or array.
        params: 4-element list [a, b, c, d] where:
                a = left foot
                b = left plateau start (membership = 1)
                c = right plateau end (membership = 1)
                d = right foot
    Returns:
        Membership degree in [0.0, 1.0].
    """
    a, b, c, d = params
    x_arr = np.asarray(x, dtype=float)
    y = np.zeros_like(x_arr)
    
    # Left slope
    if a != b:
        left_mask = (x_arr >= a) & (x_arr < b)
        y[left_mask] = (x_arr[left_mask] - a) / (b - a)
        
    # Plateau
    plateau_mask = (x_arr >= b) & (x_arr <= c)
    y[plateau_mask] = 1.0
    
    # Right slope
    if c != d:
        right_mask = (x_arr > c) & (x_arr <= d)
        y[right_mask] = (d - x_arr[right_mask]) / (d - c)
        
    y = np.clip(y, 0.0, 1.0)
    if np.isscalar(x):
        return float(y.item())
    return y


class LinguisticVariable:
    """Represents a linguistic variable (e.g. 'hazard_density') with fuzzy terms."""

    def __init__(self, name: str, terms: Dict[str, List[float]]):
        self.name = name
        self.terms = terms  # term_name -> parameters

    def membership(self, term: str, value: float) -> float:
        """Calculate membership degree of a value for a specific linguistic term."""
        if term not in self.terms:
            raise KeyError(f"Term '{term}' not defined for variable '{self.name}'")
        params = self.terms[term]
        if len(params) == 3:
            return float(trimf(value, params))
        elif len(params) == 4:
            return float(trapmf(value, params))
        else:
            raise ValueError(f"Invalid parameter count {len(params)} for term '{term}'")

    def fuzzify(self, value: float) -> Dict[str, float]:
        """Compute membership degrees across all terms for this variable."""
        return {term: self.membership(term, value) for term in self.terms}


class FuzzySystemVariables:
    """Manager for all linguistic variables in the SafePath safety system."""

    def __init__(self, config: Dict[str, Dict[str, List[float]]] = None):
        self.config = config or DEFAULT_MEMBERSHIP_CONFIG
        self.variables: Dict[str, LinguisticVariable] = {
            var_name: LinguisticVariable(var_name, terms)
            for var_name, terms in self.config.items()
        }

    def get_variable(self, name: str) -> LinguisticVariable:
        return self.variables[name]

    def fuzzify_all(self, numeric_features: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Fuzzify a dictionary of numeric features."""
        fuzzified = {}
        for var_name, value in numeric_features.items():
            if var_name in self.variables:
                fuzzified[var_name] = self.variables[var_name].fuzzify(value)
        return fuzzified

