
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate
import pandas as pd
from sklearn.model_selection import learning_curve, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import optuna
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

PATH = 'dataset.csv'
N_TRIALS = 5

class Dataset:
    def __init__(self, path=PATH, balance_data=False, test_size=0.3):
        self.path = path
        self.balance_data = balance_data
        self.test_size = test_size
        self.X_train, self.X_test, self.y_train, self.y_test = self.prepare_data()

    def prepare_data(self):
        dataset = pd.read_csv(self.path)
        dataset.drop(['id'], axis=1, inplace=True)

        # suddivido in train e test
        X = dataset.drop(['diagnosis'], axis=1)
        y = dataset['diagnosis'].map({'B': 0, 'M': 1})

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        # Modifica qui: solo transform per X_test
        X_test_scaled = scaler.transform(X_test)

        if self.balance_data:
            smote = SMOTE(random_state=42)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)

        return X_train_scaled, X_test_scaled, y_train, y_test

class Model:
    def __init__(self, dataset, n_trials=N_TRIALS, cv=5):
        self.dataset = dataset
        self.n_trials = n_trials
        self.cv = cv

    def objective(self, trial):
        model = RandomForestClassifier(
             n_estimators=trial.suggest_int('n_estimators', 20, 50),
             max_depth=trial.suggest_int('max_depth', 1, 5), 
             min_samples_split=trial.suggest_int('min_samples_split', 7, 10),  
             min_samples_leaf=trial.suggest_int('min_samples_leaf', 7, 15),  
             max_features=trial.suggest_float('max_features', 0.1, 1.0),  
             random_state=42,
             class_weight='balanced',
        )

        scores = cross_val_score(model, self.dataset.X_train, self.dataset.y_train, cv=self.cv, n_jobs=1)
        return np.mean(scores)

    def run(self, complexity_param, param_range):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.n_trials)

        best_model = RandomForestClassifier(**study.best_params)
        best_model.fit(self.dataset.X_train, self.dataset.y_train)
        model_pred = best_model.predict(self.dataset.X_test)

        accuracy = accuracy_score(self.dataset.y_test, model_pred)
        precision = precision_score(self.dataset.y_test, model_pred, average='weighted')
        recall = recall_score(self.dataset.y_test, model_pred, average='weighted')
        f1 = f1_score(self.dataset.y_test, model_pred, average='weighted')

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("Classification Report:")
        print(classification_report(self.dataset.y_test, model_pred))

        train_scores = []
        test_scores = []

        # Calcola le score per ogni valore di parametro
        for param_value in param_range:
            params = study.best_params.copy()
            params[complexity_param] = param_value
            model = RandomForestClassifier(**params, random_state=42)
            
            scores = cross_validate(model, self.dataset.X_train, self.dataset.y_train, cv=self.cv, scoring='accuracy', return_train_score=True)
            train_scores.append(np.mean(scores['train_score']))
            test_scores.append(np.mean(scores['test_score']))

        # Creazione della curva di apprendimento
        plt.figure(figsize=(10, 6))
        plt.title(f"Curva di Apprendimento - {complexity_param} vs. Score")
        plt.xlabel(complexity_param)
        plt.ylabel("Score")
        
        plt.plot(param_range, train_scores, 'o-', color="r", label="Training score")
        plt.plot(param_range, test_scores, 'o-', color="g", label="Test score")
        plt.ylim(0.8, 1)

        # Linea di riempimento tra le curve
        plt.fill_between(param_range, train_scores, test_scores, alpha=0.1, color="g", label='Score Fill')
        plt.legend(loc="best")
        plt.grid()

        plt.axvline(x=study.best_params[complexity_param], color='blue', linestyle='--', label='Best Param Value')
        plt.legend(loc='best')
        
        plt.show()
        plt.close()
        

if __name__ == '__main__':
    dataset = Dataset()
    model = Model(dataset)

    complexity_param = 'max_depth'
    param_range = range(1, 10)

    model.run(complexity_param=complexity_param, param_range=param_range)