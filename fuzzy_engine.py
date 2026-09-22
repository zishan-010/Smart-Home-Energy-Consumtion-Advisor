"""
fuzzy_engine.py
----------------
A genuine Mamdani fuzzy inference system (built with scikit-fuzzy) that scores
the "energy waste risk" of a household's appliance usage pattern.

Pipeline: fuzzification -> rule evaluation (min/max inference) -> centroid defuzzification.

Inputs (crisp, numeric):
    load_w        : appliance load in watts            (0 - 5000)
    duration_hr   : daily usage duration in hours       (0 - 24)
    occupancy_n   : number of people home during usage  (0 - 10)

Output:
    risk score (0 - 100): higher = more energy is likely being wasted
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ----------------------------------------------------------------------
# Build the fuzzy control system once at import time (cheap to reuse).
# ----------------------------------------------------------------------

# ---- Antecedents (inputs) with their universes of discourse ----
load = ctrl.Antecedent(np.arange(0, 5001, 1), "load")
duration = ctrl.Antecedent(np.arange(0, 25, 1), "duration")
occupancy = ctrl.Antecedent(np.arange(0, 11, 1), "occupancy")

# ---- Consequent (output) ----
risk = ctrl.Consequent(np.arange(0, 101, 1), "risk")

# ---- Membership functions (fuzzification) ----
load["low"] = fuzz.trimf(load.universe, [0, 0, 1500])
load["medium"] = fuzz.trimf(load.universe, [1000, 2500, 4000])
load["high"] = fuzz.trimf(load.universe, [3000, 5000, 5000])

duration["short"] = fuzz.trimf(duration.universe, [0, 0, 6])
duration["moderate"] = fuzz.trimf(duration.universe, [4, 9, 14])
duration["long"] = fuzz.trimf(duration.universe, [10, 24, 24])

occupancy["low"] = fuzz.trimf(occupancy.universe, [0, 0, 2])
occupancy["medium"] = fuzz.trimf(occupancy.universe, [1, 3, 5])
occupancy["high"] = fuzz.trimf(occupancy.universe, [4, 10, 10])

risk["low"] = fuzz.trimf(risk.universe, [0, 0, 35])
risk["medium"] = fuzz.trimf(risk.universe, [25, 50, 75])
risk["high"] = fuzz.trimf(risk.universe, [65, 100, 100])

# ---- Rule base (expert knowledge + extra coverage rules) ----
rules = [
    # Original high-risk rules
    ctrl.Rule(load["high"] & duration["long"], risk["high"]),
    ctrl.Rule(load["high"] & occupancy["low"], risk["high"]),
    ctrl.Rule(load["medium"] & duration["long"] & occupancy["low"], risk["high"]),
    ctrl.Rule(load["high"] & duration["long"] & occupancy["high"], risk["high"]),

    # Original low-risk rules
    ctrl.Rule(load["low"] & duration["short"], risk["low"]),
    ctrl.Rule(load["low"] & occupancy["high"], risk["low"]),
    ctrl.Rule(load["medium"] & duration["short"], risk["low"]),

    # Original medium-risk rules
    ctrl.Rule(duration["moderate"] & load["medium"], risk["medium"]),
    ctrl.Rule(occupancy["high"] & duration["moderate"] & load["medium"], risk["medium"]),
    ctrl.Rule(load["high"] & duration["short"] & occupancy["high"], risk["medium"]),
    ctrl.Rule(load["low"] & duration["long"] & occupancy["low"], risk["medium"]),

    # ---------- Extra coverage rules (prevents empty output) ----------
    ctrl.Rule(load["medium"] & duration["long"], risk["high"]),
    ctrl.Rule(load["high"] & duration["moderate"], risk["high"]),
    ctrl.Rule(load["medium"] & duration["moderate"] & occupancy["low"], risk["medium"]),
    ctrl.Rule(load["low"] & duration["moderate"], risk["low"]),
    ctrl.Rule(load["medium"] & duration["short"] & occupancy["medium"], risk["low"]),
    ctrl.Rule(load["high"] & duration["short"] & occupancy["medium"], risk["medium"]),
    ctrl.Rule(load["low"] & duration["long"] & occupancy["medium"], risk["medium"]),
    ctrl.Rule(load["medium"] & duration["long"] & occupancy["medium"], risk["high"]),
    ctrl.Rule(load["high"] & duration["moderate"] & occupancy["medium"], risk["high"]),
    ctrl.Rule(load["low"] & duration["short"] & occupancy["medium"], risk["low"]),
]

_energy_ctrl_system = ctrl.ControlSystem(rules)


def compute_risk(load_w: float, duration_hr: float, occupancy_n: float) -> dict:
    """
    Run the fuzzy inference system on one set of crisp inputs.

    Returns a dict with:
        - score: defuzzified crisp risk score (0-100)
        - band: human label ("Low" / "Medium" / "High")
        - memberships: fuzzification detail for each input variable (for transparency/UI)
        - inputs: the clipped crisp values that were used
    """
    # Clip into valid universe ranges so extreme LLM-extracted values don't break inference
    load_w = float(np.clip(load_w, 0, 5000))
    duration_hr = float(np.clip(duration_hr, 0, 24))
    occupancy_n = float(np.clip(occupancy_n, 0, 10))

    sim = ctrl.ControlSystemSimulation(_energy_ctrl_system)
    sim.input["load"] = load_w
    sim.input["duration"] = duration_hr
    sim.input["occupancy"] = occupancy_n
    sim.compute()

    # -------- Safe fallback if no rules fired --------
    if "risk" not in sim.output:
        # Simple weighted heuristic (keeps the app from crashing)
        load_norm = load_w / 5000.0
        dur_norm = duration_hr / 24.0
        occ_norm = 1.0 - (occupancy_n / 10.0)   # lower occupancy → higher waste risk
        score = 100.0 * (0.45 * load_norm + 0.40 * dur_norm + 0.15 * occ_norm)
        score = float(np.clip(score, 0, 100))
    else:
        score = float(sim.output["risk"])

    if score < 35:
        band = "Low"
    elif score < 65:
        band = "Medium"
    else:
        band = "High"

    memberships = {
        "load": {
            "low": float(fuzz.interp_membership(load.universe, load["low"].mf, load_w)),
            "medium": float(fuzz.interp_membership(load.universe, load["medium"].mf, load_w)),
            "high": float(fuzz.interp_membership(load.universe, load["high"].mf, load_w)),
        },
        "duration": {
            "short": float(fuzz.interp_membership(duration.universe, duration["short"].mf, duration_hr)),
            "moderate": float(fuzz.interp_membership(duration.universe, duration["moderate"].mf, duration_hr)),
            "long": float(fuzz.interp_membership(duration.universe, duration["long"].mf, duration_hr)),
        },
        "occupancy": {
            "low": float(fuzz.interp_membership(occupancy.universe, occupancy["low"].mf, occupancy_n)),
            "medium": float(fuzz.interp_membership(occupancy.universe, occupancy["medium"].mf, occupancy_n)),
            "high": float(fuzz.interp_membership(occupancy.universe, occupancy["high"].mf, occupancy_n)),
        },
    }

    return {
        "score": round(score, 1),
        "band": band,
        "memberships": memberships,
        "inputs": {"load_w": load_w, "duration_hr": duration_hr, "occupancy_n": occupancy_n},
    }


if __name__ == "__main__":
    # quick manual sanity check
    print(compute_risk(4200, 18, 1))   # expect high risk
    print(compute_risk(600, 3, 5))     # expect low risk
    print(compute_risk(2200, 9, 4))    # expect medium risk