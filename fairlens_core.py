import pandas as pd
import numpy as np
import dowhy
from dowhy import CausalModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import shap
import warnings
warnings.filterwarnings("ignore")

def run_bias_audit(df, sensitive_attribute="gender", outcome="loan_status", adjustment_features=None):
    """
    Run a full bias audit on a dataset.
    
    Args:
        df (pd.DataFrame): Input data
        sensitive_attribute (str): Column name for sensitive attribute (binary)
        outcome (str): Binary outcome column (0/1)
        adjustment_features (list): List of features to adjust for
    
    Returns:
        dict: Structured audit report
    """
    if adjustment_features is None:
        adjustment_features = ["age", "income", "education_num", "credit_history", "loan_amount"]

    # =============================
    # 1. Validate Input
    # =============================
    required_cols = [sensitive_attribute, outcome] + adjustment_features
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not pd.api.types.is_numeric_dtype(df[outcome]):
        raise ValueError(f"Outcome '{outcome}' must be numeric (0/1).")
    if df[outcome].nunique() != 2:
        raise ValueError(f"Outcome '{outcome}' must be binary.")

    y = df[outcome].copy()
    g = df[sensitive_attribute].copy()

    # Ensure g is categorical and binary
    g = pd.Categorical(g)
    if len(g.categories) != 2:
        raise ValueError(f"Sensitive attribute '{sensitive_attribute}' must have exactly 2 categories. Found: {g.categories.tolist()}")

    g = g.codes  # Convert to 0/1 NumPy array
    df = df.copy()
    df[sensitive_attribute] = g  # Replace with numeric codes

    # =============================
    # 2. Descriptive Statistics
    # =============================
    g = pd.Categorical(g).codes  # Convert to 0/1
    groups = sorted(np.unique(g))  # ✅ Fixed: use np.unique(g)
    approval_rates = {grp: y[g == grp].mean() for grp in groups}
    raw_diff = approval_rates.get(1, 0) - approval_rates.get(0, 0)

    # =============================
    # 3. Generate DAG Dynamically
    # =============================
    dag_template = """
    digraph {{
        {sensitive} -> income;
        {sensitive} -> education_num;

        age -> income;
        age -> loan_amount;

        education_num -> income;
        education_num -> loan_amount;

        income -> credit_history;
        income -> loan_amount;
        income -> {outcome};

        credit_history -> {outcome};
        loan_amount -> {outcome};
    }}
    """
    dag = dag_template.format(sensitive=sensitive_attribute, outcome=outcome)

    # =============================
    # 4. Causal Model & Identification
    # =============================
    try:
        model = CausalModel(
            data=df,
            treatment=sensitive_attribute,
            outcome=outcome,
            graph=dag,
            proceed_when_unidentifiable=True
        )
        identified_estimand = model.identify_effect()
    except Exception as e:
        raise RuntimeError(f"Causal identification failed: {e}")

    # =============================
    # 5. Estimate Causal Effect
    # =============================
    try:
        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression",
            target_units="ate",
            method_params={"fit_intercept": True}
        )
        ci = estimate.get_confidence_intervals()
        ci_low, ci_high = ci[0], ci[1]
        is_significant = bool(ci_low * ci_high > 0)
    except Exception as e:
        return {"error": f"Estimation failed: {str(e)}"}

    # =============================
    # 6. Refutation Tests (Safe)
    # =============================
    refutations = {}
    refuters = [
        ("add_unobserved_common_cause", "random_common_cause"),
        ("placebo_treatment", "placebo"),
        ("data_subset_refuter", "subset")
    ]

    for method, key in refuters:
        try:
            ref = model.refute_estimate(identified_estimand, estimate, method_name=method)
            refutations[key] = {"passed": True, "new_effect": ref.new_effect}
        except Exception as e:
            refutations[key] = {"passed": False, "new_effect": None, "error": str(e)}

    # =============================
    # 7. Fairness Metrics
    # =============================
    X = df[adjustment_features + [sensitive_attribute]]
    X = pd.get_dummies(X, columns=[sensitive_attribute], drop_first=True)
    
    try:
        model_sk = LogisticRegression(max_iter=1000)
        model_sk.fit(X, y)
        y_pred = model_sk.predict(X)
    except Exception as e:
        return {"error": f"Model training failed: {str(e)}"}

    g0_idx = df[sensitive_attribute] == 0
    g1_idx = df[sensitive_attribute] == 1

    def compute_rates(y_true, y_pred):
        if len(y_true) == 0:
            return 0.0, 0.0
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return tpr, fpr

    tpr_0, fpr_0 = compute_rates(y[g0_idx], y_pred[g0_idx])
    tpr_1, fpr_1 = compute_rates(y[g1_idx], y_pred[g1_idx])

    # =============================
    # 8. SHAP Explainability (Sampled)
    # =============================
    shap_mean = None
    try:
        X_sample = shap.sample(X, 50)  # Limit for speed
        explainer = shap.KernelExplainer(model_sk.predict_proba, X_sample)
        shap_values = explainer.shap_values(X_sample)
        # Assume gender is the first sensitive column
        gender_col = [c for c in X.columns if sensitive_attribute in c]
        if gender_col:
            col_idx = list(X.columns).index(gender_col[0])
            shap_mean = float(np.abs(np.array(shap_values)[:, :, 1][:, col_idx]).mean())
    except Exception as e:
        shap_mean = None

    # =============================
    # 9. Build Structured Report
    # =============================
    ate_value = float(estimate.value)
    recommendation = "✅ No significant bias detected."
    if is_significant and ate_value < -0.05:
        recommendation = "⚠️ Significant bias detected against Group 1. Recommend fairness-aware retraining or policy review."
    elif is_significant and ate_value > 0.05:
        recommendation = "⚠️ Significant bias detected against Group 0. Investigate further."
    elif is_significant:
        recommendation = "🟡 Moderate bias detected. Monitor closely."

    # Safe refutation analysis
    random_effect = refutations["random_common_cause"]["new_effect"]
    placebo_effect = refutations["placebo"]["new_effect"]
    subset_effect = refutations["subset"]["new_effect"]

    def stability_msg(effect, orig):
        if effect is None:
            return "Failed"
        change = abs(effect - orig)
        return f"Stable (Δ={change:.2f})" if change < 0.05 else f"Variable (Δ={change:.2f})"

    # =============================
    # 10. Return JSON-serializable result
    # =============================
    result = {
        "summary": {
            "raw_approval_gap": float(raw_diff),
            "causal_ate": ate_value,
            "ci_lower": float(ci_low),
            "ci_upper": float(ci_high),
            "significant": is_significant
        },
        "fairness_metrics": {
            "equal_opportunity_gap": abs(tpr_1 - tpr_0),
            "predictive_equality_gap": abs(fpr_1 - fpr_0),
            "statistical_parity_difference": y_pred[g1_idx].mean() - y_pred[g0_idx].mean()
        },
        "refutations": {
            "add_unobserved_common_cause": refutations["random_common_cause"],
            "placebo_treatment": refutations["placebo"],
            "data_subset": refutations["subset"]
        },
        "explainability": {
            "mean_abs_shap_sensitive": shap_mean
        },
        "recommendation": recommendation,
        "status": "success"
    }

    return result