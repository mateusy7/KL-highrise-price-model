import numpy as np

from preprocess.preprocessing import preprocess_linear, preprocess_tree

from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import Ridge
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def make_model_linear(d=1, reg=1):
    
    preprocessing = preprocess_linear(d)
    target_transform = TransformedTargetRegressor(
        regressor=Ridge(alpha=reg),
        transformer=FunctionTransformer(np.log, inverse_func=np.exp)
    )

    model = Pipeline([
        ("preprocess", preprocessing),
        ("model", target_transform)
    ])

    return model

def make_model_tree(depth=None, min_samples=2):
    
    preprocessing = preprocess_tree()
    target_transform = TransformedTargetRegressor(
        regressor=DecisionTreeRegressor(max_depth=depth, min_samples_split=min_samples, random_state=1),
        transformer=FunctionTransformer(np.log, inverse_func=np.exp)
    )

    model = Pipeline([
        ("preprocess", preprocessing),
        ("model", target_transform)
    ])

    return model

def make_model_forest(n=100, depth=None, min_samples=2):
    
    preprocessing = preprocess_tree()
    target_transform = TransformedTargetRegressor(
        regressor=RandomForestRegressor(n_estimators=n,
                                        max_depth=depth,
                                        min_samples_split=min_samples,
                                        random_state=1),
        transformer=FunctionTransformer(np.log, inverse_func=np.exp)
    )

    model = Pipeline([
        ("preprocess", preprocessing),
        ("model", target_transform)
    ])

    return model