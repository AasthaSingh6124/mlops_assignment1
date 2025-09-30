# train.py
from sklearn.tree import DecisionTreeRegressor
from misc import load_data, get_X_y, split_data, build_pipeline, cross_validate_model, train_and_evaluate

def main():
    df = load_data()
    X, y = get_X_y(df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeRegressor(random_state=42)
    # Decision trees don't strictly need scaling, so scale=False
    pipeline = build_pipeline(model, scale=False)

    cv_mse = cross_validate_model(X, y, pipeline, cv=5)
    test_mse = train_and_evaluate(pipeline, X_train, X_test, y_train, y_test)

    print(f"DecisionTree - CV mean MSE: {cv_mse:.4f}")
    print(f"DecisionTree - Holdout MSE: {test_mse:.4f}")

if __name__ == "__main__":
    main()
