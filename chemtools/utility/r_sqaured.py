import numpy as np

def centered_r_squared(y, y_pred):
    """
    Calcola l'R^2 centrato.
    
    Parametri:
    y : array
        Valori osservati
    y_pred : array
        Valori predetti dal modello
    
    Restituisce:
    r_squared : float
        Coefficiente di determinazione R^2 centrato
    """
    # Calcolo del numeratore e del denominatore della formula per R^2 centrato
    numerator = np.sum(np.square(y_pred - np.mean(y)))
    denominator = np.sum(np.square(y - np.mean(y)))
    
    # Calcolo di R^2 centrato
    return numerator / denominator

def uncentered_r_squared(y, y_pred):
    """
    Calcola l'R^2 non centrato.
    
    Parametri:
    y : array
        Valori osservati
    y_pred : array
        Valori predetti dal modello
    
    Restituisce:
    r_squared : float
        Coefficiente di determinazione R^2 non centrato
    """
    # Calcolo del numeratore e del denominatore della formula per R^2 non centrato
    numerator = np.sum(np.square(y_pred))
    denominator = np.sum(np.square(y))
    
    # Calcolo di R^2 non centrato
    return numerator / denominator


def uncentered_adjusted_r_squared(y, y_pred, k):
    """
    Calcola l'adjusted R^2 non centrato.
    
    Parametri:
    y : array
        Valori osservati
    y_pred : array
        Valori predetti dal modello
    k : int
        Numero di predittori nel modello
    
    Restituisce:
    adjusted_r_squared : float
        Coefficiente di determinazione adjusted R^2 non centrato
    """
    # Calcolo dell'adjusted R^2 non centrato utilizzando la formula sopra riportata
    n = len(y)
    return 1 - (1 - uncentered_r_squared(y, y_pred)) * (n - 1) / (n - k)