# utils_guardrails.py
import numpy as np
import pandas as pd

def clip_percent(s: pd.Series) -> pd.Series:
    return s.clip(lower=0, upper=100)

def enforce_positive(s: pd.Series, epsilon: float = 1e-9) -> pd.Series:
    return s.clip(lower=epsilon)

def winsorize_series(s: pd.Series, lower_q=0.01, upper_q=0.99):
    lo, hi = s.quantile(lower_q), s.quantile(upper_q)
    return s.clip(lo, hi)

def safe_log(s: pd.Series, epsilon: float = 1e-9) -> pd.Series:
    return np.log(enforce_positive(s, epsilon))

def inv_log(s: pd.Series) -> pd.Series:
    return np.exp(s)

def safe_logit(s: pd.Series, eps=1e-6) -> pd.Series:
    # Expect s in [0,100] or [0,1]; normalize if needed
    if s.max() > 1.0 + 1e-9:
        s = s / 100.0
    s = s.clip(eps, 1-eps)
    return np.log(s/(1-s))

def inv_logit(s: pd.Series) -> pd.Series:
    p = 1/(1+np.exp(-s))
    return 100.0*p  # return 0..100
