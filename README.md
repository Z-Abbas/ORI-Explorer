# ORI-Explorer
A unified cell-specific tool for Origin of Replication sites Prediction by Feature Fusion

![block_diagram](https://user-images.githubusercontent.com/80881943/227442401-63f37866-b005-460e-94db-ab8edba1e8fc.png)


## Specifications
- Python 3.7
- tensorflow 2.4.1
- keras 2.4.3
- numpy 1.18.5
- pandas 1.2.4

## Analysis
### SHAP
SHapley Additive exPlanation is usually termed as SHAP is a widespread utility to find out the features which are most consequential during the prediction of a sample by ML or DL models. We used it to show the feature importance of the top 20 features as shown in the figure below.

![shap](https://user-images.githubusercontent.com/80881943/234735967-077d6efc-e68f-42d3-bfbc-0ed5074fedf4.png)


### Cross-specie
We assessed the applicability of a specie or cell-specific model to other cells by cross-specie testing.


![cross](https://user-images.githubusercontent.com/80881943/234737096-6245b417-f3d7-4c6d-9c96-c82483c4a9c8.png)
