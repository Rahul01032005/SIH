import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

class RiskModel:
    def __init__(self, algorithm: str = "random_forest", random_state: int = 42):
        self.model = LogisticRegression(max_iter=1000, random_state=random_state) if algorithm == "logistic_regression" else RandomForestClassifier(n_estimators=150, random_state=random_state)
        self.feature_names: list[str] = []
    def train(self, x, y): self.feature_names = list(x.columns); self.model.fit(x, y); return self
    def predict(self, x): return self.model.predict(x)
    def predict_proba(self, x): return self.model.predict_proba(x)[:, 1]
    def evaluate(self, x, y) -> dict:
        predicted = self.predict(x); result = {"precision": precision_score(y, predicted, zero_division=0), "recall": recall_score(y, predicted, zero_division=0), "f1": f1_score(y, predicted, zero_division=0), "confusion_matrix": confusion_matrix(y, predicted).tolist()}
        if len(np.unique(y)) > 1: result["roc_auc"] = roc_auc_score(y, self.predict_proba(x))
        return result
