def degrees_of_freedom2(n_observation, coeff):
    """
    Calcola i gradi di libertà per una regressione.

    Parametri:
    n_observation: int
        Il numero totale di osservazioni (campioni).
    coeff: array-like
        Il vettore dei coefficienti calcolati con la regressione.

    Restituisce:
    int
        I gradi di libertà per la regressione.
    """
    
    # numero di coefficienti
    p = len(coeff)

    return n_observation - p - 1

def degrees_of_freedom(X):
    """
    Calcola i gradi di libertà dei residui per una regressione OLS data una matrice di dati X.

    Parametri:
    X : array_like
        Matrice delle variabili indipendenti. Deve avere dimensione (n, p), dove n è il numero di osservazioni e p è il numero di predittori.

    Restituisce:
    df_resid : int
        Gradi di libertà dei residui per la regressione OLS.
    """
    return X.shape[0] - X.shape[1]