import numpy as np


def prediction_band(number_data, x, x_mean, y_pred_orig, SSxx, t_two):
    """
    Calcola la banda di predizione per una regressione lineare.

    Parametri:
    number_data (int): Numero totale di osservazioni.
    x (array): Array contenente i valori di x per cui si vuole calcolare la banda di predizione.
    x_mean (float): Media dei valori di x.
    y_pred_orig (array): Array contenente i valori predetti di y per i valori di x in input.
    SSxx (float): Somma dei quadrati delle deviazioni di X dalla sua media.
    t_two (float): Valore critico della distribuzione t di Student con gradi di libertà pari al numero di osservazioni meno il numero di parametri stimati.

    Restituisce:
    PI_Y_upper (array): Array contenente gli estremi superiori della banda di predizione per ogni valore x in input.
    PI_Y_lower (array): Array contenente gli estremi inferiori della banda di predizione per ogni valore x in input.

    Esempio:

            # Definisci i parametri in input
            number_data = 100
            x = np.array([...])
            x_mean = np.mean(x)
            y_pred_orig = np.array([...])
            SSxx = np.sum((x - x_mean) ** 2)
            t_two = stats.t.ppf(1 - 0.05 / 2, number_data - 2)

            # Chiama la funzione prediction_band passando i parametri in input
            PI_Y_upper, PI_Y_lower = prediction_band(number_data, x, x_mean, y_pred_orig, SSxx, t_two)

            # Visualizza gli estremi superiori e inferiori della banda di predizione
            print(PI_Y_upper)
            print(PI_Y_lower)

    """

    # Calcola la banda di predizione superiore e inferiore utilizzando operazioni vettoriali
    PI_Y_upper = y_pred_orig + t_two * np.sqrt(
        1 + 1 / number_data + (x - x_mean) ** 2 / SSxx
    )
    PI_Y_lower = y_pred_orig - t_two * np.sqrt(
        1 + 1 / number_data + (x - x_mean) ** 2 / SSxx
    )

    return PI_Y_upper, PI_Y_lower
