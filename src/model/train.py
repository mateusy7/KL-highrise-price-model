from pathlib import Path
from model.models import make_model_linear, make_model_tree, make_model_forest
from sklearn.model_selection import GridSearchCV
import joblib

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)

def train_model(X_train, y_train, model_name="forest_v1.pkl"):
    
    print("Training model...")
    model = make_model_forest(n=250, depth=32, min_samples=2)

    model.fit(X_train, y_train)

    model_path = MODEL_DIR / model_name
    joblib.dump(model, model_path)

    print(f"Model saved to {model_path}\n")

    return model

def tune_model(X_train, y_train, scoring='neg_root_mean_squared_error'):
    n_estimators_list = [50, 100, 250, 500]
    max_depth_list = [8, 16, 32, None]
    min_samples_split_list = [2, 5, 10, 20]

    model = make_model_forest()
    max_depth_name = "model__regressor__max_depth"
    min_samples_split_name = "model__regressor__min_samples_split"
    n_estimators_name = "model__regressor__n_estimators"

    params = {
    n_estimators_name:n_estimators_list,
    max_depth_name:max_depth_list,
    min_samples_split_name:min_samples_split_list
    }

    search = GridSearchCV(model, params, scoring=scoring)
    search.fit(X_train, y_train)

    best_params = search.best_params_
    max_depth = best_params[max_depth_name]
    min_samples_split = best_params[min_samples_split_name]
    n_estimators = best_params[n_estimators_name]

    return search.best_estimator_, n_estimators, max_depth, min_samples_split