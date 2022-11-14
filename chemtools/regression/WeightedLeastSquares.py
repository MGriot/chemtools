import numpy as np

class WeightedLeastSquares:
    """Weighted Least Squares class"""
    def __init__(self):
        pass

    def regression(self, x, y):
        """Date due matrici X e Y esegue la regressione

        Args:
            x (matrix): _description_, serve una matrice contenente conentenente, oltre ai valori di x, anche una colonna formata da soli 1 (per ottenere l'intercetta)
                        esempio:
                        array([[1., 0.],
                               [1., 1.],
                               [1., 2.],
                               [1., 3.],
                               [1., 4.],
                               [1., 5.],
                               [1., 6.]])
            y (matrix): variabili dipendenti

        Returns:
            beta (array): stimatori dei minimi quadrati.
        """
        try:
            beta = np.dot((np.linalg.inv(np.dot(x.T,x))), np.dot(x.T,y))
        except Exception:
            beta = np.dot((np.linalg.pinv(np.dot(x.T,x))), np.dot(x.T,y))
        self.beta=beta
        
        return beta
