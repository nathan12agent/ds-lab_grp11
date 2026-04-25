from screensight.models.logi          import get_model as get_lr,  get_param_grid as get_lr_grid
from screensight.models.svm           import get_model as get_svm, get_param_grid as get_svm_grid
from screensight.models.randomforrest import get_model as get_rf,  get_param_grid as get_rf_grid
from screensight.models.gradient      import get_model as get_gb,  get_param_grid as get_gb_grid
from screensight.models.xgbooost      import get_model as get_xgb, get_param_grid as get_xgb_grid

MODEL_REGISTRY = {
    "Logistic Regression": get_lr,
    "SVM":                 get_svm,
    "Random Forest":       get_rf,
    "Gradient Boosting":   get_gb,
    "XGBoost":             get_xgb,
}

PARAM_GRID_REGISTRY = {
    "Logistic Regression": get_lr_grid,
    "SVM":                 get_svm_grid,
    "Random Forest":       get_rf_grid,
    "Gradient Boosting":   get_gb_grid,
    "XGBoost":             get_xgb_grid,
}