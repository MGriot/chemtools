import matplotlib.pyplot as plt

from chemtools.utility import directory_creator
from chemtools.utility import when_date


def matplotlib_savefig(savefig=False, DPI=None, output="output", name="", fig_format="png", transparent_background=False):
    """
    Funzione che permette di standardizzare il processo di salvaggio dei 
    grafici ottenuti con matplotlib.

    Args:
        savefig (bool, optional): Permette di scegliere se salvare (True) il grafico o mostrarlo solamente (False). Defaults to False.
        DPI (_type_, optional): Valore di DPI con cui generare il grafico che verrà poi salvato. Defaults to None or 300.
        output (str, optional): Cartella o percorso in cui si vogliono salvare tutti i file generati. Defaults to "output".
        name (str, optional): Nome specifico che si vuole dare al grafico. Defaults to "".
        fig_format (str, optional): Estensione dell'immagine che viene generata. Defaults to "png".
        transparent_background (bool, optional): Permette di settare il background dell'immagine trasparente (True) o no (False). Defaults to False.

    Returns:
        _type_: Rimanda un anteprima del grafico che si è generato, sia che si salvi l'immagine che no.
    """
    if savefig == True:
        directory_creator(output)
        if DPI is None:
            plt.savefig(f'./{output}/{when_date()}_{name}_plot.{fig_format}',
                        transparent=transparent_background, bbox_inches='tight')
        if DPI != None:
            plt.savefig(f'./{output}/{when_date()}_{name}_plot.{fig_format}',
                        transparent=transparent_background, bbox_inches='tight', dpi=DPI)
    return plt.show()
