from src.config import COST_FEATURES, TIME_FEATURES
from src.dataset import build_supervised_dataset, load_and_validate
from src.training import make_pipeline

def test_real_dataset_and_no_july_feature_leakage():
    raw = load_and_validate(); data = build_supervised_dataset()
    assert len(raw) == 7590 and set(data.snapshot_month.unique()) <= {"2026-04","2026-05","2026-06"}
    assert "revised_cost" not in COST_FEATURES and "revised_completion_date" not in TIME_FEATURES

def test_unseen_categories_do_not_crash():
    data = build_supervised_dataset().dropna(subset=["cost_overrun_by_july"])
    model = make_pipeline(COST_FEATURES,"logistic").fit(data.iloc[:100][COST_FEATURES],data.iloc[:100].cost_overrun_by_july.astype(int))
    row = data.iloc[[101]][COST_FEATURES].copy(); row["state"] = "Unseen state"
    assert len(model.predict_proba(row)) == 1
