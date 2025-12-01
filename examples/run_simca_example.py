import numpy as np
from chemtools.classification.simca import SIMCA

def generate_data():
    """Generates a synthetic dataset with three distinct classes."""
    np.random.seed(42)
    # Class A: Centered around [10, 12], with some correlation
    cov_a = [[1, 0.8], [0.8, 1]]
    class_a = np.random.multivariate_normal([10, 12], cov_a, 50)

    # Class B: Centered around [20, 22], with different correlation
    cov_b = [[1.5, -0.6], [-0.6, 1.5]]
    class_b = np.random.multivariate_normal([20, 22], cov_b, 50)
    
    # Class C: Centered around [15, 18], more spread out
    cov_c = [[2, 0], [0, 2]]
    class_c = np.random.multivariate_normal([15, 18], cov_c, 50)

    # Combine into a single training dataset
    X_train = np.vstack([class_a, class_b, class_c])
    y_train = np.array(['Class A'] * 50 + ['Class B'] * 50 + ['Class C'] * 50)

    # Generate new points for prediction
    X_new = np.array([
        [10.5, 12.5],  # Should be Class A
        [20.2, 21.8],  # Should be Class B
        [15.1, 17.9],  # Should be Class C
        [18, 15],      # Might be an outlier or belong to C
        [12, 20]       # Likely an outlier
    ])
    
    print("Generated 150 training samples and 5 new samples for prediction.")
    return X_train, y_train, X_new

def run_simca():
    """
    An example demonstrating the use of the SIMCA classifier.
    """
    # 1. Generate synthetic data
    X_train, y_train, X_new = generate_data()

    # 2. Initialize and fit the SIMCA model
    # We will use 1 principal component for each class model.
    print("\nFitting SIMCA model with n_components=1...")
    simca_model = SIMCA(n_components=1, alpha=0.05)
    simca_model.fit(X_train, y_train)

    # Print a summary of the fitted model
    print("\n--- SIMCA Model Summary ---")
    print(simca_model.summary)
    print("---------------------------\n")

    # 3. Predict the class of new, unknown samples
    print("Predicting classes for 5 new samples...")
    predictions = simca_model.predict(X_new)

    # 4. Display the results
    print("\n--- Prediction Results ---")
    for i, p in enumerate(predictions):
        sample_info = f"Sample {i+1} (data: {np.round(X_new[i], 2)}):"
        if not p:
            class_assignment = "Is an OUTLIER (does not belong to any class)."
        else:
            class_assignment = f"Belongs to class(es): {', '.join(p)}"
        
        print(f"{sample_info.ljust(35)} {class_assignment}")
    print("--------------------------")

if __name__ == "__main__":
    run_simca()
