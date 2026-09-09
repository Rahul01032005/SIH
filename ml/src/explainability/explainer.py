def feature_importance(model) -> list[dict]:
    """Return model signals, not causal explanations, from a saved sklearn pipeline."""
    if hasattr(model, "named_steps"):
        estimator = model.named_steps["model"]
        names = model.named_steps["preprocess"].get_feature_names_out()
    else:
        estimator = getattr(model, "model", model); names = getattr(model, "feature_names", [])
    values = getattr(estimator, "feature_importances_", None)
    if values is None and hasattr(estimator, "coef_"): values = abs(estimator.coef_[0])
    if values is None: return []
    return [{"feature": name, "importance": round(float(value), 4)} for name, value in sorted(zip(names, values), key=lambda pair: pair[1], reverse=True)]

def explain_top_factors(model, limit: int = 3) -> list[str]:
    return [f"{item['feature']} is an important model signal associated with predicted risk ({item['importance']:.3f})." for item in feature_importance(model)[:limit]]
