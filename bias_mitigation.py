# bias_mitigation.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference
)
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

def mitigate_bias(df, sensitive_attribute="gender", outcome="loan_status", method="threshold"):
    """
    Apply bias mitigation to a loan/hiring dataset.
    """
    try:
        # Validate input
        if sensitive_attribute not in df.columns:
            raise ValueError(f"Sensitive attribute '{sensitive_attribute}' not in data")
        if outcome not in df.columns:
            raise ValueError(f"Outcome '{outcome}' not in data")

        # Encode education if present
        df = df.copy()
        if "education" in df.columns:
            le = LabelEncoder()
            df["education"] = le.fit_transform(df["education"])

        # Features and labels
        X = df.drop([outcome], axis=1)
        y = df[outcome]
        sf = df[sensitive_attribute]

        # Ensure binary sensitive attribute
        sf = pd.Categorical(sf)
        if len(sf.categories) != 2:
            raise ValueError(f"Sensitive attribute must be binary. Got {len(sf.categories)} categories.")
        sf = sf.codes  # Convert to 0/1

        # Split
        X_train, X_test, y_train, y_test, sf_train, sf_test = train_test_split(
            X, y, sf, test_size=0.3, random_state=42, stratify=y
        )

        # Remove sensitive attribute from features
        if sensitive_attribute in X_train.columns:
            X_train = X_train.drop(columns=[sensitive_attribute])
            X_test = X_test.drop(columns=[sensitive_attribute])

        # Train base model
        model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)

        # Base predictions
        y_pred_base = model.predict(X_test)
        y_proba_base = model.predict_proba(X_test)[:, 1]

        # Evaluate base fairness
        dp_base = demographic_parity_difference(y_test, y_pred_base, sensitive_features=sf_test)
        eo_base = equalized_odds_difference(y_test, y_pred_base, sensitive_features=sf_test)
        acc_base = accuracy_score(y_test, y_pred_base)
        f1_base = f1_score(y_test, y_pred_base)
        bacc_base = balanced_accuracy_score(y_test, y_pred_base)

        # Apply mitigation
        if method == "threshold":
            postprocessor = ThresholdOptimizer(
                estimator=model,
                constraints="demographic_parity",
                objective="balanced_accuracy_score",
                prefit=True
            )
            postprocessor.fit(X_train, y_train, sensitive_features=sf_train)
            y_pred_mitigated = postprocessor.predict(X_test, sensitive_features=sf_test)

        elif method == "exponentiated_gradient":
            mitigator = ExponentiatedGradient(
                estimator=lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, verbose=-1),
                constraints=DemographicParity()
            )
            mitigator.fit(X_train, y_train, sensitive_features=sf_train)
            y_pred_mitigated = mitigator.predict(X_test)
        else:
            return {"error": "Method must be 'threshold' or 'exponentiated_gradient'"}

        # Evaluate mitigated fairness
        dp_mit = demographic_parity_difference(y_test, y_pred_mitigated, sensitive_features=sf_test)
        eo_mit = equalized_odds_difference(y_test, y_pred_mitigated, sensitive_features=sf_test)
        acc_mit = accuracy_score(y_test, y_pred_mitigated)
        f1_mit = f1_score(y_test, y_pred_mitigated)
        bacc_mit = balanced_accuracy_score(y_test, y_pred_mitigated)

        # Improvement
        dp_improvement = dp_base - dp_mit
        eo_improvement = eo_base - eo_mit
        acc_change = acc_mit - acc_base

        return {
            "status": "success",
            "method": method,
            "before_mitigation": {
                "accuracy": acc_base,
                "f1_score": f1_base,
                "balanced_accuracy": bacc_base,
                "demographic_parity_difference": dp_base,
                "equalized_odds_difference": eo_base
            },
            "after_mitigation": {
                "accuracy": acc_mit,
                "f1_score": f1_mit,
                "balanced_accuracy": bacc_mit,
                "demographic_parity_difference": dp_mit,
                "equalized_odds_difference": eo_mit
            },
            "improvement": {
                "demographic_parity_improvement": dp_improvement,
                "equalized_odds_improvement": eo_improvement,
                "accuracy_change": acc_change
            },
            "recommendation": (
                "Bias significantly reduced." if abs(dp_mit) < 0.1
                else "Partial reduction. Consider data preprocessing or hybrid methods."
            )
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }