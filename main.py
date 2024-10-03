import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import completeness_score, f1_score, homogeneity_score, precision_score, recall_score, silhouette_score, v_measure_score
from sklearn.preprocessing import StandardScaler


from pgmpy.estimators import K2Score, HillClimbSearch,BayesianEstimator
from pgmpy.models import BayesianNetwork, BayesianModel
from pgmpy.inference import VariableElimination
import warnings

from networkx.drawing.nx_pydot import graphviz_layout


from funzioni import *

warnings.filterwarnings('ignore')


# %%% Visualizzazione del Dataset %%%% 
dataset = pd.read_csv('dataset.csv')
NaN_Values=dataset.isnull().sum()
print(NaN_Values)
stampa_info(dataset)

# %%%% Eliminiamo colonne superflue e con 450+ valori Null%%%%
colonna_da_eliminare = ['id']
dataset = dataset.drop(colonna_da_eliminare, axis=1)
print('Numero di righe presenti nel Dataset: ', dataset.shape[0])
print('Numero di colonne presenti nel Dataset: ', dataset.shape[1])
col_discrete = [col for col in dataset.columns if (dataset[col].dtype == 'object')]
col_continue = [col for col in dataset.columns if col not in col_discrete]

print('\nLe colonne di tipologia discreta sono:\n')
print(col_discrete)
print('\n')
print('Le colonne di tipologia continua sono:\n')
print(col_continue)
print('\n\n')

#%%% Distribuzione dei casi di tumori benigni e maligni nel dataset %%%

    # Calcola il conteggio dei casi di tumore benigno e maligno
conteggio = dataset['diagnosis'].value_counts()
    
    # Prepara i dati per il grafico a torta
labels = conteggio.index
sizes = conteggio.values
colors = ['#3CB371', '#FF0000']  # Verde per i tumori benigni, rosso per i tumori maligni    

    # Crea il grafico a torta
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.axis('equal')  # Rende il grafico a torta circolare
plt.title('Distribuzione dei casi di tumore')
    
    # Mostra il grafico
plt.show()


#%%% Correlazione tra i dati %%%

visualizza_distribuzione_conteggio(dataset[['perimeter_mean','diagnosis']], 
                                   'Distribuzione casi di tumori rispetto livelli di perimetro medio del tumore:')

visualizza_distribuzione_conteggio(dataset[['area_mean','diagnosis']], 
                                   'Distribuzione casi di tumori rispetto livelli di area media del tumore:')

visualizza_distribuzione_conteggio(dataset[[ 'concavity_mean','diagnosis']], 
                                   'Distribuzione casi di tumori rispetto alla concavità media del tumore:')

# %%%% Visualizzazione dell'Heatmap %%%%
dataset_numeric=dataset.drop(col_discrete,axis=1)
plt.figure(figsize=(13, 13))
sns.heatmap(dataset_numeric.corr(), annot=True, linewidth=1.7, fmt='0.2f', cmap=sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True))
plt.show()
plt.close()


# %%%% Calcolo degli autovalori %%%%

X = dataset.drop(['diagnosis'], axis=1)
y = dataset['diagnosis']

df_utile = X.copy()
df_norm, df_stan = scala_dati(col_continue, df_utile)
ds_clustering = df_stan.copy()

i = 2
pca_test = None
while i < 30:
    pca_test = PCA(i)
    pca_test.fit_transform(ds_clustering)
    i += 1

    # Imposta il formato di stampa senza notazione esponenziale
np.set_printoptions(precision=4, suppress=True)

print('Autovalori:')
print(pca_test.explained_variance_)
print('\n\n')

plt.title('Scree Plot:')
plt.plot(pca_test.explained_variance_, marker='o', color='red')
plt.xlabel('Numero Autovalori:')
plt.ylabel('Grandezza Autovalori:')
plt.show()
plt.close()
# Ottieni i vettori dei carichi delle componenti principali
component_loadings = pca_test.components_

# Associa i carichi delle componenti alle colonne originali
component_names = df_stan.columns
component_loadings_df = pd.DataFrame(component_loadings, columns=component_names)

# Stampa i carichi delle componenti
print('Carichi delle componenti:')
print(component_loadings_df)

# Visualizza un grafico a barre dei carichi delle componenti
plt.figure(figsize=(12, 8))
plt.imshow(component_loadings_df, cmap='coolwarm', aspect='auto')
plt.xticks(range(len(component_names)), component_names, rotation=90)
plt.yticks(range(len(component_names)), component_names)
plt.colorbar(label='Carico')
plt.xlabel('Variabili')
plt.ylabel('Componenti')
plt.title('Carichi delle componenti')
plt.show()

# %%%% Clustering %%%%

# Determina il numero ottimale di cluster utilizzando il metodo del gomito
# Calcola l'inerzia per diversi valori di k
inertia = []
k_values = range(1, 30)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_stan)
    inertia.append(kmeans.inertia_)

# Traccia il grafico dell'inerzia rispetto al numero di cluster
plt.plot(k_values, inertia, 'bx-')
plt.xlabel('Numero di cluster')
plt.ylabel('Inerzia')
plt.title('Metodo del gomito')
plt.show()

ssd = []
poss_numero_clusters = [2,3,4,5,6,7,8]
pca = PCA(6)
df_kmed = pca.fit_transform(ds_clustering)

for num_clusters in poss_numero_clusters:
    kmedoids = KMedoids(n_clusters=num_clusters, method='pam', max_iter=100, init='k-medoids++', random_state=1)
    kmedoids.fit(df_kmed)
    ssd.append(kmedoids.inertia_)

    media_silhouette = silhouette_score(df_kmed, kmedoids.labels_)
    print('Con n_clusters={0}, il valore di silhouette {1}'.format(num_clusters, media_silhouette))

print('\n\n')
plt.title('Curva a gomito:')
plt.plot(ssd)
plt.grid()
plt.show()
plt.close()

kmedoids = KMedoids(n_clusters=2, method='pam', max_iter=100, init='k-medoids++', random_state=1)
label = kmedoids.fit_predict(df_kmed)
etichette_kmed = np.unique(label)
df_kmed = np.array(df_kmed)


plt.figure(figsize=(9, 9))
plt.title('Clustering con k-Medoids: ')

# Definisci i colori
colors = ['red', 'green']  # Rosso per 0, verde per 1

for k in etichette_kmed:
    plt.scatter(df_kmed[label == k, 0], df_kmed[label == k, 1], label=k, c=colors[k])

plt.scatter(kmedoids.cluster_centers_[:, 0], kmedoids.cluster_centers_[:, 1], s=200, c='k', label='Medoide')
plt.legend()
plt.show()
plt.close()

ds_clustering['cluster'] = label

print('\n\nValutazione:\n')
print('Omogeneità  : ', homogeneity_score(y, kmedoids.labels_))
print('Completezza : ', completeness_score(y, kmedoids.labels_))
print('V_measure   : ', v_measure_score(y, kmedoids.labels_))

#%%%Classificazione:
# Classificazione:

classificatori = []
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=54)
Xn_train, Xn_test, Xs_train, Xs_test = scala_dati(col_continue, X_train, X_test)

risultati_Knn = []
for k in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=k,metric='minkowski',p=2)
    valutazioni = cross_val_score(knn, Xn_train, y_train, cv=5, scoring='accuracy')
    risultati_Knn.append(valutazioni.mean())
val_x = [k for k in range(1, 20)]
plt.plot(val_x, risultati_Knn, color='g')
plt.xticks(ticks=val_x, labels=val_x)
plt.grid()
plt.show()
plt.close()


train_scores = []
test_scores = []
k_values = range(1, 21)  # Valori di k da 1 a 20

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2)
    knn.fit(X_train, y_train)
    train_score = knn.score(X_train, y_train)
    test_score = knn.score(X_test, y_test)
    train_scores.append(train_score)
    test_scores.append(test_score)

    # Visualizza i punteggi di training e test in funzione del valore di k
plt.plot(k_values, train_scores, label='Training Score', color=colors[0])
plt.plot(k_values, test_scores, label='Test Score', color=colors[1])
plt.xlabel('Valore di k')
plt.ylabel('Accuracy')
plt.title('Curva di apprendimento')
plt.legend()
plt.show()


train_scores = []
test_scores = []
C_values = [1]  # Valori di C
gamma_values = [10]  # Valori di gamma

for C in C_values:
    for gamma in gamma_values:
        svm = SVC(C=C, gamma=gamma)
        svm.fit(X_train, y_train)
        train_preds = svm.predict(X_train)
        test_preds = svm.predict(X_test)
        train_score = accuracy_score(y_train, train_preds)
        test_score = accuracy_score(y_test, test_preds)
        train_scores.append(train_score)
        test_scores.append(test_score)

# Visualizza i punteggi di training e test
plt.bar(['Train', 'Test'], [train_scores[0], test_scores[0]])
plt.ylabel('Accuracy')
plt.title('Punteggi di training e test')
plt.show()



train_scores = []
test_scores = []
criterion_values = ['entropy']  # Valori del criterio

for criterion in criterion_values:
    dtc = DecisionTreeClassifier(criterion=criterion)
    dtc.fit(X_train, y_train)
    train_preds = dtc.predict(X_train)
    test_preds = dtc.predict(X_test)
    train_score = accuracy_score(y_train, train_preds)
    test_score = accuracy_score(y_test, test_preds)
    train_scores.append(train_score)
    test_scores.append(test_score)

# Visualizza i punteggi di training e test
plt.bar(['Train', 'Test'], [train_scores[0], test_scores[0]])
plt.ylabel('Accuracy')
plt.title('Punteggi di training e test')
plt.show()

train_scores = []
test_scores = []
n_estimators_values = [25]  # Valori di n_estimators

for n_estimators in n_estimators_values:
    rfc = RandomForestClassifier(n_estimators=n_estimators)
    rfc.fit(X_train, y_train)
    train_preds = rfc.predict(X_train)
    test_preds = rfc.predict(X_test)
    train_score = accuracy_score(y_train, train_preds)
    test_score = accuracy_score(y_test, test_preds)
    train_scores.append(train_score)
    test_scores.append(test_score)

# Visualizza i punteggi di training e test
plt.bar(['Train', 'Test'], [train_scores[0], test_scores[0]])
plt.ylabel('Accuracy')
plt.title('Punteggi di training e test')
plt.show()

#%%% CROSS VALIDATION DEL DECISION TREE CLASSIFIER

target_variable = 'diagnosis'


X = dataset.drop(target_variable, axis=1)
y = dataset[target_variable]

scaler = StandardScaler()
X = scaler.fit_transform(X)


accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []


kfold = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)


for train_index, test_index in kfold.split(X, y):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    parametri = {'criterion': ['entropy']}
    clf = RandomizedSearchCV(DecisionTreeClassifier(), parametri, scoring='f1', n_iter=3)
    clf.fit(X_train, y_train)
    

    y_pred = clf.predict(X_test)
    
   
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred, pos_label='M'))
    recall_scores.append(recall_score(y_test, y_pred, pos_label='M'))
    f1_scores.append(f1_score(y_test, y_pred, pos_label='M'))


print("Valutazioni del Decision Tree Classifier, con stratified K-Cross validation pari a 15")
print("Average Accuracy:", np.mean(accuracy_scores))
print("Average Precision:", np.mean(precision_scores))
print("Average Recall:", np.mean(recall_scores))
print("Average F1-score:", np.mean(f1_scores))
print ("+------------------------------------------------------------------------------------------+")


fold_numbers = np.arange(1, len(accuracy_scores) + 1)
plt.plot(fold_numbers, f1_scores, label='F1-score')

plt.xlabel('Fold')
plt.ylabel('Score')
plt.title('Media F1 con Decision Tree Classifier')
plt.legend()
plt.grid(True)
plt.show()


#%%% CROSS VALIDATION DEL SVC

target_variable = 'diagnosis'

X = dataset.drop(target_variable, axis=1)
y = dataset[target_variable]

scaler = StandardScaler()
X = scaler.fit_transform(X)


accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []


kfold = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)


for train_index, test_index in kfold.split(X, y):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
   
   
    clf = SVC()
    clf.fit(X_train, y_train)
    

    y_pred = clf.predict(X_test)
    

    accuracy_scores.append(accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred, pos_label='M'))
    recall_scores.append(recall_score(y_test, y_pred, pos_label='M'))
    f1_scores.append(f1_score(y_test, y_pred, pos_label='M'))


print("Valutazioni del SVC, con stratified K-Cross validation pari a 15")
print("Average Accuracy:", np.mean(accuracy_scores))
print("Average Precision:", np.mean(precision_scores))
print("Average Recall:", np.mean(recall_scores))
print("Average F1-score:", np.mean(f1_scores))
print ("+------------------------------------------------------------------------------------------+")


fold_numbers = np.arange(1, len(accuracy_scores) + 1)
plt.plot(fold_numbers, f1_scores, label='F1-score')

plt.xlabel('Fold')
plt.ylabel('Score')
plt.title('Media F1 con SVC')
plt.legend()
plt.grid(True)
plt.show()

#%%% CROSS VALIDATION DEL KNN


target_variable = 'diagnosis'


X = dataset.drop(target_variable, axis=1)
y = dataset[target_variable]

scaler = StandardScaler()
X = scaler.fit_transform(X)


accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []


kfold = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)


for train_index, test_index in kfold.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
   
    clf = KNeighborsClassifier(n_neighbors=3,metric='minkowski',p=2)
    clf.fit(X_train, y_train)
    
   
    y_pred = clf.predict(X_test)
    
    
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred, pos_label='M'))
    recall_scores.append(recall_score(y_test, y_pred, pos_label='M'))
    f1_scores.append(f1_score(y_test, y_pred, pos_label='M'))


print("Valutazioni del KNN, con stratified K-Cross validation pari a 15")
print("Average Accuracy:", np.mean(accuracy_scores))
print("Average Precision:", np.mean(precision_scores))
print("Average Recall:", np.mean(recall_scores))
print("Average F1-score:", np.mean(f1_scores))
print ("+------------------------------------------------------------------------------------------+")



fold_numbers = np.arange(1, len(accuracy_scores) + 1)
plt.plot(fold_numbers, f1_scores, label='F1-score')

plt.xlabel('Fold')
plt.ylabel('Score')
plt.title('Media F1 con KNN')
plt.legend()
plt.grid(True)
plt.show()


#%%% CROSS VALIDATION DEL RANDOM FOREST 

target_variable = 'diagnosis'


X = dataset.drop(target_variable, axis=1)
y = dataset[target_variable]

scaler = StandardScaler()
X = scaler.fit_transform(X)


accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []


kfold = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)


for train_index, test_index in kfold.split(X, y):
   
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    
    parametri = {'criterion': ['entropy']}
    clf = RandomizedSearchCV(RandomForestClassifier(), parametri, scoring='f1', n_iter=7)
    clf.fit(X_train, y_train)
    
    
    y_pred = clf.predict(X_test)
    
    
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred, pos_label='M'))
    recall_scores.append(recall_score(y_test, y_pred, pos_label='M'))
    f1_scores.append(f1_score(y_test, y_pred, pos_label='M'))
print("Valutazioni del Random Forest Classifier, con stratified K-Cross validation pari a 15")
print("Average Accuracy:", np.mean(accuracy_scores))
print("Average Precision:", np.mean(precision_scores))
print("Average Recall:", np.mean(recall_scores))
print("Average F1-score:", np.mean(f1_scores))
print ("+------------------------------------------------------------------------------------------+")


fold_numbers = np.arange(1, len(accuracy_scores) + 1)
plt.plot(fold_numbers, f1_scores, label='F1-score')

plt.xlabel('Fold')
plt.ylabel('Score')
plt.title('Media F1 con Random Forest Classifier')
plt.legend()
plt.grid(True)
plt.show()

#%%% Rete Bayesiana %%%
# Rete Bayesiana
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.estimators import HillClimbSearch, K2Score, BayesianEstimator
from pgmpy.models import BayesianModel, BayesianNetwork
from pgmpy.inference import VariableElimination
from networkx.drawing.nx_agraph import graphviz_layout

# Carica il dataset
dataset = pd.read_csv("dataset.csv")

# Prendi un campione casuale dello 0.5% dei dati
dataset_reduced = dataset.sample(frac=0.005, random_state=42)
df_RBayes = dataset_reduced.copy()

# Rimuovi la colonna "id" e mappa la variabile 'diagnosis'
df_RBayes = df_RBayes.drop("id", axis=1)
df_RBayes['diagnosis'] = df_RBayes['diagnosis'].map({'B': 0, 'M': 1})

# Funzione per convertire le colonne float in interi, ma con arrotondamento
def converti_float_in_interi(dataset):
    for colonna in dataset.columns:
        if dataset[colonna].dtype == 'float64':
            dataset[colonna] = dataset[colonna].round(0).astype(int)
    return dataset

# Converti le colonne float in int dove applicabile sul campione ridotto
df_RBayes = converti_float_in_interi(df_RBayes)

# Ricerca del modello con HillClimbSearch
k2 = K2Score(df_RBayes)
hc_k2 = HillClimbSearch(df_RBayes)

# Stima del miglior modello di rete bayesiana
modello_k2 = hc_k2.estimate(scoring_method=k2)

# Creazione della rete bayesiana basata sugli archi stimati
rete_bayesiana = BayesianNetwork(modello_k2.edges())

# Fit della rete bayesiana
rete_bayesiana.fit(df_RBayes, n_jobs=1)

# Stampa i nomi dei nodi
print("Nodi della rete bayesiana:")
for node in rete_bayesiana.nodes():
    print(node)

# Stampa degli archi nella rete bayesiana
print("\nArchi nella rete bayesiana:")
for edge in rete_bayesiana.edges():
    print(edge)

# Funzione per visualizzare la rete bayesiana
def visualizza_rete_bayesiana(nodi, archi):
    grafo = nx.DiGraph()
    grafo.add_nodes_from(nodi)
    grafo.add_edges_from(archi)

    plt.figure(figsize=(15, 10))
    
    try:
        pos = graphviz_layout(grafo, prog='dot')
    except:
        print("Attenzione: pygraphviz o pydot non è installato. Usa un layout alternativo.")
        pos = nx.spring_layout(grafo, seed=42)

    nx.draw_networkx(grafo, pos, node_color='lightblue', node_size=3000, alpha=0.9, 
                     arrows=True, arrowstyle='->', arrowsize=20, font_size=12, 
                     font_family='sans-serif', edge_color='gray', font_color='black')

    plt.title("Rete Bayesiana", fontsize=16)
    plt.axis('off')
    plt.show()

# Visualizza la rete bayesiana
nodi = ['diagnosis', 'concavity_mean', 'radius_mean', 'compactness_se', 'symmetry_mean', 'perimeter_mean',
         'texture_se', 'fractal_dimension_se', 'compactness_worst', 'concavity_se', 'concave points_se',
         'concavity_worst', 'perimeter_se', 'area_worst', 'compactness_mean', 'fractal_dimension_mean',
         'area_se', 'area_mean', 'concave points_worst', 'concave points_mean', 'smoothness_mean',
         'radius_worst', 'symmetry_worst', 'smoothness_se', 'symmetry_se', 'perimeter_worst', 'texture_worst',
         'smoothness_worst', 'texture_mean', 'fractal_dimension_worst', 'radius_se']

archi = [('diagnosis', 'concavity_mean'), ('concavity_mean', 'concavity_se'), ('concavity_mean', 'concave points_se'), ('concavity_mean', 'concavity_worst'), ('radius_mean', 'compactness_se'), ('radius_mean', 'symmetry_mean'), ('compactness_se', 'area_se'), ('compactness_se', 'area_mean'), ('perimeter_mean', 'texture_se'), ('perimeter_mean', 'fractal_dimension_se'), ('perimeter_mean', 'compactness_worst'), ('concavity_se', 'concave points_worst'), ('concave points_se', 'concave points_mean'), ('concave points_se', 'smoothness_mean'), ('concavity_worst', 'perimeter_worst'), ('concavity_worst', 'radius_worst'), ('perimeter_se', 'area_worst'), ('perimeter_se', 'compactness_mean'), ('perimeter_se', 'fractal_dimension_mean'), ('area_worst', 'fractal_dimension_worst'), ('area_worst', 'texture_worst'), ('area_worst', 'radius_se'), ('radius_worst', 'symmetry_worst'), ('radius_worst', 'perimeter_se'), ('radius_worst', 'smoothness_se'), ('radius_worst', 'symmetry_se'), ('radius_worst', 'perimeter_worst'), ('symmetry_worst', 'radius_mean'), ('perimeter_worst', 'smoothness_mean'), ('texture_worst', 'smoothness_worst'), ('texture_worst', 'texture_mean'), ('smoothness_worst', 'perimeter_mean')]

# Chiamata alla funzione per visualizzare il grafico
visualizza_rete_bayesiana(nodi, archi)


# Aggiungi le variabili al modello
modello_bayesiano = BayesianModel(rete_bayesiana.edges())

# Aggiorna il modello con le nuove CPD usando il campione ridotto
modello_bayesiano.fit(df_RBayes, estimator=BayesianEstimator, prior_type='BDeu', equivalent_sample_size=10)

# Esempio di inferenza sulla rete bayesiana
inferenza = VariableElimination(modello_bayesiano)

# Stampa i valori limite (massimo e minimo) per ogni variabile del modello
for variable in modello_bayesiano.nodes():
    cpd = modello_bayesiano.get_cpds(variable)
    min_value = cpd.values.min()
    max_value = cpd.values.max()
    print(f"Valori limite per la variabile '{variable}':")
    print(f"Minimo: {min_value}")
    print(f"Massimo: {max_value}")
    print("\n")

# Controllo se smoothness_mean è nel modello
if 'smoothness_mean' not in modello_bayesiano.nodes():
    print("Attenzione: 'smoothness_mean' non è presente nel modello.")

# Esempio di query per la diagnosi maligna
maligno = inferenza.query(variables=['diagnosis'], evidence={
    'radius_mean': 17.99,
    'texture_mean': 10.38,
    'perimeter_mean': 122.8,
    'area_mean': 1001,
    'smoothness_mean': 0.1184,  # Controlla se questa variabile è nel modello
    'compactness_mean': 0.2776,
    'concavity_mean': 0.300,
    'concave points_mean': 0.1471,
    'symmetry_mean': 0.2419,
    'fractal_dimension_mean': 0.07871,
    'radius_se': 1.095,
    'texture_se': 0.9053,
    'perimeter_se': 8.589,
    'area_se': 153.4,
    'smoothness_se': 0.006399,
    'compactness_se': 0.04904,
    'concavity_se': 0.05373,
    'concave points_se': 0.01587,
    'symmetry_se': 0.03003,
    'fractal_dimension_se': 0.006193,
    'radius_worst': 25.38,
    'texture_worst': 17.33,
    'perimeter_worst': 184.6,
    'area_worst': 2019,
    'smoothness_worst': 0.1622,
    'compactness_worst': 0.6656,
    'concavity_worst': 0.7119,
    'concave points_worst': 0.2654,
    'symmetry_worst': 0.4601,
    'fractal_dimension_worst': 0.1189
})

print('\nProbabilità per una donna di avere un tumore maligno al seno: ')
print(maligno, '\n')

# Esempio di query per la diagnosi benigna
benigno = inferenza.query(variables=['diagnosis'], evidence={
    'radius_mean': 12,
    'texture_mean': 15.65,
    'perimeter_mean': 76.95,
    'area_mean': 443.3,
    'smoothness_mean': 0.09723,
    'compactness_mean': 0.07165,
    'concavity_mean': 0.04151,
    'concave points_mean': 0.01863,
    'symmetry_mean': 0.2079,
    'fractal_dimension_mean': 0.05968,
    'radius_se': 0.2271,
    'texture_se': 1.255,
    'perimeter_se': 1.441,
    'area_se': 16.16,
    'smoothness_se': 0.005969,
    'compactness_se': 0.01812,
    'concavity_se': 0.02007,
    'concave points_se': 0.007027,
    'symmetry_se': 0.01972,
    'fractal_dimension_se': 0.002607,
    'radius_worst': 13.67,
    'texture_worst': 24.9,
    'perimeter_worst': 87.78,
    'area_worst': 1603,
    'smoothness_worst': 0.1398,
    'compactness_worst': 0.2089,
    'concavity_worst': 0.3157,
    'concave points_worst': 0.1642,
    'symmetry_worst': 0.3695,
    'fractal_dimension_worst': 0.08579
})

print('\nProbabilità per una donna di avere un tumore benigno al seno: ')
print(benigno, '\n\n')
