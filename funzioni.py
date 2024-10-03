import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits import *
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import warnings




def stampa_info(dataset, num_rig=20):
    print('\nPrime ' + str(num_rig) + ' righe del dataset:\t' + (dataset.dataframeName if
          hasattr(dataset, 'dataframeName') else '') + '\n')
    print(dataset.head(num_rig))
    print('\n\nCampi descritti dalle colonne del dataset:\n', dataset.columns)
    print('\nDimensioni del dataset:\t', dataset.shape)
    print('\n\nStatistiche dataset:\n')
    print(dataset.describe(include='all'))
    print('\n\nInformazioni dataset:')
    print(dataset.info())
    print('\n\n')
    

def visualizza_distribuzione_conteggio(col, desc='Controllo distribuzione dei valori:'):
    plt.suptitle(desc)
    plt.subplots_adjust(wspace=0.6)
    
    # Definisci la palette con i colori personalizzati
    palette = ['#3CB371', '#FF0000']  # Verde per benigni, rosso per maligni
    
    # Aggiorna sns.violinplot per usare la palette
    sns.violinplot(x=col.iloc[:, 0], y=col.iloc[:, 1], hue=(col.iloc[:, 2] if len(col.columns) == 3 else None),
                   split=True, inner='quartile', palette=palette)
    
    plt.show()
    plt.close()



def scala_dati(col, dati, dati_test=None):
    norm = MinMaxScaler()
    stan = StandardScaler()

    if dati_test is None:
        dati_n = dati.copy()
        dati_s = dati.copy()

        dati_n[col] = norm.fit_transform(dati[col])
        dati_s[col] = stan.fit_transform(dati[col])

        return dati_n, dati_s

    else:

        dati_n1 = dati.copy()
        dati_n2 = dati_test.copy()
        dati_s1 = dati.copy()
        dati_s2 = dati_test.copy()

        dati_n1[col] = norm.fit_transform(dati[col])
        dati_n2[col] = norm.transform(dati_test[col])
        dati_s1[col] = stan.fit_transform(dati[col])
        dati_s2[col] = stan.transform(dati_test[col])

        return dati_n1, dati_n2, dati_s1, dati_s2



def osserva_test_modello(mod, X_test, y_test):
    pred = mod.predict(X_test)

    plt.figure(figsize=(7, 7))
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, cmap='Purples', fmt='d', cbar=False)
    plt.xlabel("Val Predetti")
    plt.ylabel("Val Reali")

    print('\n+------------------------------------------------------------------------------------------+')
    print('\n+\tRisultati test per\t ' + (mod._annotations_ if hasattr(mod, '_annotations_') else str(mod)) + ' :\n\n')
    print(classification_report(y_test, pred, zero_division=0))
    print('+------------------------------------------------------------------------------------------+\n')

    plt.show()
    plt.close()

