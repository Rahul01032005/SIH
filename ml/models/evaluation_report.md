# ML model evaluation

Selection uses F1, then PR-AUC and ROC-AUC as tie-breakers.

## Cost overrun

Selected production model: **random_forest**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| logistic | 0.8023 | 0.6137 | 0.7787 | 0.6864 | 0.8822 | 0.7729 |
| random_forest | 0.8374 | 0.6968 | 0.7339 | 0.7149 | 0.8845 | 0.7834 |

## Time overrun

Selected production model: **random_forest**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| logistic | 0.8861 | 0.9623 | 0.8905 | 0.9250 | 0.9367 | 0.9770 |
| random_forest | 0.9061 | 0.9442 | 0.9362 | 0.9402 | 0.9525 | 0.9862 |
