import numpy as np
from .PrincipalComponentAnalysis import PrincipalComponentAnalysis


# GCCA
class GeneralizedCanonicalCorrelationAnalysis(PrincipalComponentAnalysis):
    def fit(self):
        try:
            X1_meaned = self.X - np.mean(self.X, axis=0)
            X2_meaned = self.X - np.mean(self.X, axis=0)
            cov_X1 = np.cov(X1_meaned, rowvar=False)
            cov_X2 = np.cov(X2_meaned, rowvar=False)
            cov_X1X2 = np.cov(X1_meaned.T, X2_meaned.T)[
                : self.X.shape[1], self.X.shape[1] :
            ]
            self.V, self.L = np.linalg.eigh(np.dot(np.linalg.inv(cov_X1), cov_X1X2))
            self.order = np.argsort(self.V)[::-1]
            self.V_ordered = self.V[self.order]
            self.L_ordered = self.L[:, self.order]
            self.PC_index = np.array([f"PC{i+1}" for i in range(self.V.shape[0])])
        except np.linalg.LinAlgError as e:
            print(f"Error during the calculation of eigenvalues and eigenvectors: {e}")
        except Exception as e:
            print(f"Unknown error: {e}")
