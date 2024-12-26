import numpy as np
from sklearn.linear_model import Ridge


def get_matrix_input(rows, cols, prompt):
    print(prompt)
    matrix = []
    for i in range(rows):
        while True:
            try:
                row = list(map(float, input(f"Row {i + 1}: ").split()))
                if len(row) != cols:
                    raise ValueError(f"Expected {cols} values, got {len(row)}")
                matrix.append(row)
                break
            except ValueError as e:
                print(f"Invalid input: {e}")
    return np.array(matrix)


def main():
    # Collecting user inputs
    try:
        num_features = int(input("Number of features: "))
        num_samples = int(input("Number of data points: "))
        if num_features <= 0 or num_samples <= 0:
            raise ValueError("Number of features and samples must be positive integers.")
    except ValueError as e:
        print(f"Invalid input: {e}")
        return

    # Collect feature data
    data = get_matrix_input(num_samples, num_features,
                            f"Enter feature values (rows: {num_samples}, columns: {num_features}):")

    # Collect target data
    while True:
        try:
            target = list(map(float, input("Enter target values: ").split()))
            if len(target) != num_samples:
                raise ValueError(f"Expected {num_samples} target values, got {len(target)}")
            target = np.array(target)
            break
        except ValueError as e:
            print(f"Invalid input: {e}")

    # Ridge regression model
    try:
        alpha = float(input("Regularization strength (alpha): "))
        if alpha < 0:
            raise ValueError("Alpha must be a non-negative value.")
    except ValueError as e:
        print(f"Invalid input: {e}")
        return

    # Fit the model
    model = Ridge(alpha=alpha)
    model.fit(data, target)

    # Output results
    print("\nRidge Regression Results:")
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)


if __name__ == "__main__":
    main()
