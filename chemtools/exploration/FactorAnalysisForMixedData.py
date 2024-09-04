import numpy as np
from .PrincipalComponentAnalysis import PrincipalComponentAnalysis


# FAMD
class FactorAnalysisForMixedData(PrincipalComponentAnalysis):
    def fit(self):
        try:
            X_continuous_normalized = (self.X - np.mean(self.X, axis=0)) / np.std(
                self.X, axis=0
            )
            X_combined = np.hstack((X_continuous_normalized, self.X))
            self.cov_mat = np.cov(X_combined, rowvar=False)
            self.V, self.L = np.linalg.eigh(self.cov_mat)
            self.order = np.argsort(self.V)[::-1]
            self.V_ordered = self.V[self.order]
            self.L_ordered = self.L[:, self.order]
            self.PC_index = np.array([f"PC{i+1}" for i in range(self.V.shape[0])])
        except np.linalg.LinAlgError as e:
            print(f"Error during the calculation of eigenvalues and eigenvectors: {e}")
        except Exception as e:
            print(f"Unknown error: {e}")
