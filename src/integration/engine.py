
import numpy as np
import sympy as sp
from scipy.integrate import quad
from typing import Callable, Dict, Tuple, Optional

# --- MÉTODOS DE INTEGRAÇÃO ---
def riemann_esquerda(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int) -> float:
    h = (b - a) / n
    x = np.linspace(a, b - h, n)
    return float(np.sum(f(x)) * h)

def riemann_direita(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int) -> float:
    h = (b - a) / n
    x = np.linspace(a + h, b, n)
    return float(np.sum(f(x)) * h)

def riemann_ponto_medio(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int) -> float:
    h = (b - a) / n
    x = np.linspace(a + h / 2, b - h / 2, n)
    return float(np.sum(f(x)) * h)

def trapezoidal(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int) -> float:
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    return float(h * (np.sum(y) - 0.5 * (y[0] + y[-1])))

def simpson(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int) -> float:
    if n % 2 == 1:
        n += 1
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    return float((h / 3) * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2])))

METODOS: Dict[str, Callable[[Callable[[np.ndarray], np.ndarray], float, float, int], float]] = {
    "Riemann Esquerda": riemann_esquerda,
    "Riemann Direita": riemann_direita,
    "Ponto Médio": riemann_ponto_medio,
    "Trapezoidal": trapezoidal,
    "Simpson": simpson,
}

ORDEM_TEORICA = {
    "Riemann Esquerda": 1,
    "Riemann Direita": 1,
    "Ponto Médio": 2,
    "Trapezoidal": 2,
    "Simpson": 4,
}

# --- PROCESSAMENTO MATEMÁTICO ---
def parse_function(expr_str: str):
    x_sym = sp.Symbol("x", real=True)
    locals_map = {
        "x": x_sym,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
        "Abs": sp.Abs, "abs": sp.Abs, "pi": sp.pi,
    }
    expr = sp.sympify(expr_str, locals=locals_map)
    f_num = sp.lambdify(x_sym, expr, modules=["numpy"])
    return expr, f_num

def build_xcurve(a: float, b: float, pontos: int = 1400) -> np.ndarray:
    pad = 0.15 * (b - a)
    return np.linspace(a - pad, b + pad, pontos)

def safe_eval_curve(expr_str: str, a: float, b: float, pontos: int = 1400) -> Tuple[np.ndarray, np.ndarray]:
    expr, f_num = parse_function(expr_str)
    x_curve = build_xcurve(a, b, pontos)
    try:
        y = f_num(x_curve)
        y = np.array(y, dtype=float)
    except Exception:
        y = np.array([float(f_num(xx)) for xx in x_curve], dtype=float)
    y[~np.isfinite(y)] = np.nan
    return x_curve, y

def compute_reference_quad(expr_str: str, a: float, b: float, max_subdiv: int = 200) -> Tuple[Optional[float], Optional[float]]:
    try:
        _, f_num = parse_function(expr_str)
        val, err = quad(lambda t: float(f_num(t)), a, b, limit=max_subdiv)
        return float(val), float(err)
    except Exception:
        return None, None

def series_convergencia(expr_str: str, a: float, b: float, nome_metodo: str, n_max: int, step: int) -> Tuple[np.ndarray, np.ndarray]:
    _, f_num = parse_function(expr_str)
    ref, _ = compute_reference_quad(expr_str, a, b)
    if ref is None:
        return np.array([], dtype=int), np.array([], dtype=float)

    ns = np.arange(10, n_max + 1, step, dtype=int)
    fn = METODOS[nome_metodo]

    errs = []
    for nn in ns:
        v = fn(f_num, a, b, int(nn))
        errs.append(abs(ref - v))
    return ns, np.array(errs, dtype=float)

def estimate_observed_order(ns: np.ndarray, errs: np.ndarray, a: float, b: float) -> Optional[float]:
    if len(ns) < 5:
        return None
    eps = 1e-300
    h = (b - a) / ns.astype(float)
    y = np.log(np.maximum(errs, eps))
    x = np.log(h)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 5:
        return None

    p = np.polyfit(x, y, 1)[0]
    return float(p)
