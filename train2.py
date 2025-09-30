# train2.py
from sklearn.kernel_ridge import KernelRidge
from misc import load_data, get_X_y, split_data, build_pipeline, cross_validate_model, train_and_evaluate

def main():
    df = load_data()
    X, y = get_X_y(df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    # KernelRidge benefits from scaling
    model = KernelRidge(alpha=1.0, kernel='rbf')  # you can tune alpha or kernel
    pipeline = build_pipeline(model, scale=True)

    cv_mse = cross_validate_model(X, y, pipeline, cv=5)
    test_mse = train_and_evaluate(pipeline, X_train, X_test, y_train, y_test)

    print(f"KernelRidge - CV mean MSE: {cv_mse:.4f}")
    print(f"KernelRidge - Holdout MSE: {test_mse:.4f}")

if __name__ == "__main__":
    main()
