import os
from typing import Any, Dict, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.model_selection import train_test_split

app = FastAPI(title="Portfolio Rebalancer ML API")

MODEL_DIR = "models"
CLASS_MODEL_PATH = os.path.join(MODEL_DIR, "risk_clf.joblib")
REG_MODEL_PATH = os.path.join(MODEL_DIR, "bond_reg.joblib")

np.random.seed(42)

# -------------------- Pydantic schemas --------------------
class Holdings(BaseModel):
    equity: float
    mf: float
    fd: float
    bonds: float

class TrainResponse(BaseModel):
    status: str
    clf_report: Dict[str, Any]
    reg_mae: float

class RiskResponse(BaseModel):
    risk_score: float
    risk_label: Literal["Low", "Medium", "High"]

class RebalanceResponse(BaseModel):
    original: Dict[str, float]
    suggested: Dict[str, float]
    moves: Dict[str, float]
    notes: str

# -------------------- Helpers & synthetic data --------------------

def make_synthetic_dataset(n=2000, random_state=42):
    rng = np.random.RandomState(random_state)
    # generate random holdings (positive numbers)
    equity = rng.uniform(0, 200_000, size=n)
    mf = rng.uniform(0, 200_000, size=n)
    fd = rng.uniform(0, 200_000, size=n)
    bonds = rng.uniform(0, 200_000, size=n)

    df = pd.DataFrame({"equity": equity, "mf": mf, "fd": fd, "bonds": bonds})
    df["total"] = df.sum(axis=1)
    # features as percentages to be scale-invariant
    for c in ["equity", "mf", "fd", "bonds"]:
        df[c + "_pct"] = df[c] / df["total"]

    # Create a rule-based risk score (continuous)
    # weights: equity=3, mf=2, fd=0.8, bonds=0.5
    df["original_risk_score"] = (
        df["equity_pct"] * 3.0 + df["mf_pct"] * 2.0 + df["fd_pct"] * 0.8 + df["bonds_pct"] * 0.5
    )
    
    # Scale the risk score from 0-3 to 0-10
    df["risk_score"] = (df["original_risk_score"] / 3.0) * 10.0

    # create labels - using original thresholds for consistency
    df["risk_label"] = pd.cut(df["original_risk_score"], bins=[-1, 0.9, 1.6, 3.0], labels=["Low", "Medium", "High"])

    # Define an "optimal" target bond percentage (rule-based):
    # - If High risk: target_bond_pct = current_bonds_pct + 0.25 (cap 0.8)
    # - If Medium: +0.15
    # - If Low: +0.05
    def target_bond_pct(row):
        cur = row["bonds_pct"]
        if row["risk_label"] == "High":
            t = min(cur + 0.25, 0.8)
        elif row["risk_label"] == "Medium":
            t = min(cur + 0.15, 0.7)
        else:
            t = min(cur + 0.05, 0.6)
        # Also push a fixed chunk of FD into bonds: move 20% of FD (in pct) to bonds
        t = min(t + 0.20 * row["fd_pct"], 0.95)
        return t

    df["target_bond_pct"] = df.apply(target_bond_pct, axis=1)

    # target allocation must sum to 1 -> other assets shrink proportionally
    return df


def prepare_training_data(df):
    features = df[["equity_pct", "mf_pct", "fd_pct", "bonds_pct"]].values
    clf_y = df["risk_label"].values
    reg_y = df["target_bond_pct"].values
    return features, clf_y, reg_y


def ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------- Training --------------------

def train_models(n_samples=3000, random_state=42):
    df = make_synthetic_dataset(n=n_samples, random_state=random_state)
    X, y_clf, y_reg = prepare_training_data(df)

    X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = train_test_split(
        X, y_clf, y_reg, test_size=0.2, random_state=random_state
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=random_state)
    clf.fit(X_train, y_clf_train)

    y_clf_pred = clf.predict(X_test)
    clf_report = classification_report(y_clf_test, y_clf_pred, output_dict=True)

    reg = RandomForestRegressor(n_estimators=200, random_state=random_state)
    reg.fit(X_train, y_reg_train)

    y_reg_pred = reg.predict(X_test)
    reg_mae = float(mean_absolute_error(y_reg_test, y_reg_pred))

    ensure_model_dir()
    joblib.dump(clf, CLASS_MODEL_PATH)
    joblib.dump(reg, REG_MODEL_PATH)

    return clf_report, reg_mae


# -------------------- Utilities --------------------

def load_models():
    if os.path.exists(CLASS_MODEL_PATH) and os.path.exists(REG_MODEL_PATH):
        clf = joblib.load(CLASS_MODEL_PATH)
        reg = joblib.load(REG_MODEL_PATH)
        return clf, reg
    return None, None


def compute_rule_risk_and_label(holdings: Dict[str, float]):
    eq, mf, fd, bd = holdings["equity"], holdings["mf"], holdings["fd"], holdings["bonds"]
    total = eq + mf + fd + bd
    if total <= 0:
        return 0.0, "Low"
    eq_pct, mf_pct, fd_pct, bd_pct = eq / total, mf / total, fd / total, bd / total
    # Calculate the original risk score (0-3 scale)
    original_rs = eq_pct * 3.0 + mf_pct * 2.0 + fd_pct * 0.8 + bd_pct * 0.5
    
    # Scale the risk score from 0-3 to 0-10
    # The theoretical max value is 3.0 (if 100% in equity)
    rs = (original_rs / 3.0) * 10.0
    
    # Use the original thresholds but scaled to the new range
    # Original thresholds: 0.9 and 1.6 on a 0-3 scale
    # New thresholds: 3.0 and 5.33 on a 0-10 scale
    if original_rs <= 0.9:
        lab = "Low"
    elif original_rs <= 1.6:
        lab = "Medium"
    else:
        lab = "High"
    return float(rs), lab


def ml_predict(holdings: Dict[str, float]):
    clf, reg = load_models()
    eq, mf, fd, bd = holdings["equity"], holdings["mf"], holdings["fd"], holdings["bonds"]
    total = eq + mf + fd + bd
    if total <= 0:
        raise ValueError("Total holdings must be > 0")
    features = np.array([[eq / total, mf / total, fd / total, bd / total]])

    if clf is None or reg is None:
        raise FileNotFoundError("Models not trained. Call /train first.")

    pred_label = clf.predict(features)[0]
    pred_bond_pct = float(reg.predict(features)[0])
    # ensure valid range
    pred_bond_pct = max(0.0, min(0.95, pred_bond_pct))

    return {"pred_label": str(pred_label), "pred_bond_pct": pred_bond_pct}


# -------------------- Rebalance logic --------------------

def compute_rebalance(holdings: Dict[str, float], target_bond_pct: float) -> Dict[str, Any]:
    eq, mf, fd, bd = holdings["equity"], holdings["mf"], holdings["fd"], holdings["bonds"]
    total = eq + mf + fd + bd
    cur = {"equity": eq, "mf": mf, "fd": fd, "bonds": bd}
    cur_pct = {k: v / total for k, v in cur.items()}

    # compute target bonds absolute
    target_bonds_abs = target_bond_pct * total
    delta_bonds = target_bonds_abs - bd

    moves = {"from_equity": 0.0, "from_mf": 0.0, "from_fd": 0.0}

    if delta_bonds <= 0:
        notes = "No increase in bonds required (target <= current). Consider other strategies."
        suggested = cur.copy()
        return {"original": cur, "suggested": suggested, "moves": moves, "notes": notes}

    # We will take from equities and MFs proportionally to their available amount above a minimum allocation
    # and also take a fixed fraction from FD (20% of FD) as per requirement
    take_from_fd = min(0.20 * fd, delta_bonds)
    remaining_needed = max(0.0, delta_bonds - take_from_fd)

    # available pools in eq and mf
    # define a soft minimum to keep: 5% of total in each risky asset to avoid zeroing out completely
    min_each_abs = 0.05 * total
    avail_eq = max(0.0, eq - min_each_abs)
    avail_mf = max(0.0, mf - min_each_abs)
    total_avail = avail_eq + avail_mf

    if total_avail > 0 and remaining_needed > 0:
        frac_eq = avail_eq / total_avail if total_avail > 0 else 0.5
        take_eq = min(avail_eq, remaining_needed * frac_eq)
        take_mf = min(avail_mf, remaining_needed * (1 - frac_eq))
        # if still short due to caps, take remaining from whichever has leftover
        taken = take_eq + take_mf
        shortfall = remaining_needed - taken
        if shortfall > 1e-6:
            # try to take from whichever has more avail
            if avail_eq - take_eq > avail_mf - take_mf:
                extra = min(shortfall, avail_eq - take_eq)
                take_eq += extra
                shortfall -= extra
            if shortfall > 1e-6:
                extra = min(shortfall, avail_mf - take_mf)
                take_mf += extra
                shortfall -= extra
    else:
        take_eq = 0.0
        take_mf = 0.0

    # Final check: sum of moves should be >= delta_bonds (allow small numeric slack)
    total_taken = take_from_fd + take_eq + take_mf
    if total_taken + 1e-6 < delta_bonds:
        # If insufficient, scale up proportionally from eq and mf (could also reduce target)
        scale = delta_bonds / max(total_taken, 1e-9)
        take_from_fd *= scale
        take_eq *= scale
        take_mf *= scale
        total_taken = take_from_fd + take_eq + take_mf

    # Apply moves
    suggested = {
        "equity": eq - take_eq,
        "mf": mf - take_mf,
        "fd": fd - take_from_fd,
        "bonds": bd + total_taken,
    }

    moves = {"from_equity": round(take_eq, 2), "from_mf": round(take_mf, 2), "from_fd": round(take_from_fd, 2)}
    suggested = {k: round(v, 2) for k, v in suggested.items()}

    notes = f"Moved funds into bonds: total moved = {round(total_taken,2)}"
    return {"original": {k: round(v, 2) for k, v in cur.items()}, "suggested": suggested, "moves": moves, "notes": notes}

# -------------------- API Endpoints --------------------

@app.post("/train", response_model=TrainResponse)
def train_endpoint():
    clf_report, reg_mae = train_models(n_samples=4000, random_state=42)
    return {"status": "trained", "clf_report": clf_report, "reg_mae": reg_mae}


@app.post("/predict_risk", response_model=RiskResponse)
def predict_risk_endpoint(h: Holdings):
    holdings = h.dict()
    # If models exist, use ML prediction; otherwise fall back to rule-based
    clf, reg = load_models()
    if clf is not None:
        try:
            ml = ml_predict(holdings)
            # compute a continuous risk score using rule for compatibility
            rs, _ = compute_rule_risk_and_label(holdings)
            return {"risk_score": rs, "risk_label": ml["pred_label"]}
        except Exception:
            rs, lab = compute_rule_risk_and_label(holdings)
            return {"risk_score": rs, "risk_label": lab}
    else:
        rs, lab = compute_rule_risk_and_label(holdings)
        return {"risk_score": rs, "risk_label": lab}


@app.post("/rebalance", response_model=RebalanceResponse)
def rebalance_endpoint(h: Holdings):
    holdings = h.dict()
    # Use ML to get target bond pct if possible
    clf, reg = load_models()
    if clf is not None:
        try:
            ml = ml_predict(holdings)
            target_bond_pct = ml["pred_bond_pct"]
        except Exception:
            # fallback
            _, lab = compute_rule_risk_and_label(holdings)
            # simple fallback target
            if lab == "High":
                target_bond_pct = 0.5
            elif lab == "Medium":
                target_bond_pct = 0.35
            else:
                target_bond_pct = 0.2
    else:
        _, lab = compute_rule_risk_and_label(holdings)
        if lab == "High":
            target_bond_pct = 0.5
        elif lab == "Medium":
            target_bond_pct = 0.35
        else:
            target_bond_pct = 0.2

    reb = compute_rebalance(holdings, target_bond_pct)
    notes = f"Target bonds pct = {round(target_bond_pct,3)}. " + reb["notes"]
    reb["notes"] = notes
    return reb


@app.get("/test")
def run_tests():
    # Simple built-in testcases
    tests = [
        {"equity": 80000, "mf": 10000, "fd": 5000, "bonds": 5000},
        {"equity": 10000, "mf": 5000, "fd": 70000, "bonds": 10000},
        {"equity": 200000, "mf": 150000, "fd": 50000, "bonds": 10000},
        {"equity": 0, "mf": 0, "fd": 100000, "bonds": 0},
    ]
    results = []
    for t in tests:
        try:
            r = rebalance_endpoint(Holdings(**t))
        except Exception as e:
            r = {"error": str(e)}
        results.append({"input": t, "result": r})
    return {"tests_run": len(tests), "results": results}


# Auto-train on startup if models missing (helpful for quick testing)
if __name__ == "__main__":
    if not (os.path.exists(CLASS_MODEL_PATH) and os.path.exists(REG_MODEL_PATH)):
        print("Models not found, training...")
        train_models()

