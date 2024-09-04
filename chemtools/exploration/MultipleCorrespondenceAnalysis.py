import numpy as np
from .PrincipalComponentAnalysis import PrincipalComponentAnalysis


# MCA
class MultipleCorrespondenceAnalysis(PrincipalComponentAnalysis):
    def fit(self):
        try:
            B = np.dot(self.X.T, self.X) / self.X.shape[0]
            self.V, self.L = np.linalg.eigh(B)
            self.order = np.argsort(self.V)[::-1]
            self.V_ordered = self.V[self.order]
            self.L_ordered = self.L[:, self.order]
            self.PC_index = np.array([f"PC{i+1}" for i in range(self.V.shape[0])])
        except np.linalg.LinAlgError as e:
            print(f"Error during the calculation of eigenvalues and eigenvectors: {e}")
        except Exception as e:
            print(f"Unknown error: {e}")
