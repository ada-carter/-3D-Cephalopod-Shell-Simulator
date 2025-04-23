import numpy as np
import json
import os


def fit_septa_indices(septa_indices, theta_steps, turns, degree=3):
    """
    Fit a polynomial to septa positions given indices.
    returns polynomial coefficients for theta vs index.
    """
    # build theta array
    full = np.linspace(0, 2 * np.pi * turns, theta_steps + 1)
    thetas = full[1:]
    # ensure indices list
    idx = np.array(septa_indices, dtype=int)
    x = np.arange(len(idx))
    y = thetas[idx]
    coeffs = np.polyfit(x, y, degree)
    return coeffs.tolist()


def save_septa_equation(coeffs, name, folder=".septa_equations"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.json")
    with open(path, 'w') as f:
        json.dump({"coeffs": coeffs}, f)
    return path


def load_septa_equation(path):
    """Load septa equation coefficients from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data.get('coeffs', [])


def fit_shell_growth(last_radii, theta_steps, turns):
    """
    Fit logarithmic spiral: log(r)=log(r0)+k*theta
    returns [k, log(r0)]
    """
    # build theta array
    full = np.linspace(0, 2 * np.pi * turns, theta_steps + 1)
    thetas = full[1:]
    # sample radii for first vertex in each ring
    r = last_radii
    logr = np.log(r)
    coeffs = np.polyfit(thetas[:len(r)], logr, 1)
    return coeffs.tolist()


def save_shell_equation(coeffs, name, folder=".shell_equations"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.json")
    with open(path, 'w') as f:
        json.dump({"coeffs": coeffs}, f)
    return path


def load_shell_equation(path):
    """Load shell growth equation coefficients from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data.get('coeffs', [])
