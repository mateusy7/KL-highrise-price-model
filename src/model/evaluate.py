from sklearn.metrics import root_mean_squared_error

def evaluate_model(model_fitted, X_train, y_train, X_test, y_test, scoring='root_mean_squared_error'):

    if scoring == 'root_mean_squared_error':
        error_func = root_mean_squared_error

    y_hat = model_fitted.predict(X_train)
    error_train = error_func(y_train, y_hat)

    y_hat2 = model_fitted.predict(X_test)
    error_test = error_func(y_test, y_hat2)

    print(f"Train error: {error_train:0.2f}, Test error: {error_test:0.2f}")

    return error_train, error_test