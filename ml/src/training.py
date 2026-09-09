import json
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .config import COST_FEATURES, MODELS_PATH, RANDOM_STATE, TIME_FEATURES

def make_pipeline(features, model_name):
    categorical = [x for x in features if x in {"state", "agency", "snapshot_month"}]
    numeric = [x for x in features if x not in categorical]
    preprocess = ColumnTransformer([("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric), ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical)])
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE) if model_name == "logistic" else RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    return Pipeline([("preprocess", preprocess), ("model", model)])

def metric_dict(y, probability):
    predicted = (probability >= .5).astype(int)
    result = {"accuracy": accuracy_score(y,predicted), "precision": precision_score(y,predicted,zero_division=0), "recall": recall_score(y,predicted,zero_division=0), "f1": f1_score(y,predicted,zero_division=0), "roc_auc": roc_auc_score(y,probability) if len(np.unique(y)) == 2 else None, "pr_auc": average_precision_score(y,probability), "confusion_matrix": confusion_matrix(y,predicted,labels=[0,1]).tolist()}
    return {key: round(value,4) if isinstance(value,float) else value for key,value in result.items()}

def train_target(data, target, features):
    subset = data.dropna(subset=[target]); splitter = GroupShuffleSplit(n_splits=1,test_size=.25,random_state=RANDOM_STATE)
    train_index, test_index = next(splitter.split(subset,subset[target],groups=subset.project_id))
    train, test = subset.iloc[train_index], subset.iloc[test_index]
    assert not set(train.project_id).intersection(test.project_id)
    results, pipelines = {}, {}
    for name in ("logistic","random_forest"):
        pipeline = make_pipeline(features,name).fit(train[features],train[target].astype(int))
        results[name] = metric_dict(test[target].astype(int),pipeline.predict_proba(test[features])[:,1]); pipelines[name] = pipeline
        joblib.dump(pipeline,MODELS_PATH/f"{target.replace('_by_july','')}_{name}.joblib")
    selected = max(results,key=lambda name:(results[name]["f1"],results[name]["pr_auc"],results[name]["roc_auc"] or -1))
    joblib.dump(pipelines[selected],MODELS_PATH/f"{target.replace('_by_july','')}_production.joblib")
    return {"selected_model":selected,"metrics":results,"train_projects":int(train.project_id.nunique()),"test_projects":int(test.project_id.nunique()),"features":features}

def write_evaluation_report(result):
    lines = ["# ML model evaluation", "", "Selection uses F1, then PR-AUC and ROC-AUC as tie-breakers.", ""]
    for target, details in result.items():
        lines.extend([f"## {target.title()} overrun", "", f"Selected production model: **{details['selected_model']}**", "", "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |", "|---|---:|---:|---:|---:|---:|---:|"])
        for name, values in details["metrics"].items(): lines.append(f"| {name} | {values['accuracy']:.4f} | {values['precision']:.4f} | {values['recall']:.4f} | {values['f1']:.4f} | {values['roc_auc']:.4f} | {values['pr_auc']:.4f} |")
        lines.append("")
    (MODELS_PATH/"evaluation_report.md").write_text("\n".join(lines),encoding="utf-8")

def train_all(data):
    MODELS_PATH.mkdir(parents=True,exist_ok=True)
    result = {"cost":train_target(data,"cost_overrun_by_july",COST_FEATURES),"time":train_target(data,"time_overrun_by_july",TIME_FEATURES)}
    (MODELS_PATH/"metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    write_evaluation_report(result)
    return result
