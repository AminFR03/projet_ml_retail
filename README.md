# Analyse Comportementale Clientèle Retail - Atelier Machine Learning

## Description du Projet
Ce projet est développé dans le cadre d'un atelier pratique de Machine Learning visant à analyser le comportement des clients d'une entreprise e-commerce de cadeaux. L'objectif est d'explorer une base de données complexe, de préparer les données (nettoyage, feature engineering), de modéliser le comportement (classification pour la prédiction de churn), et enfin de déployer le meilleur modèle via une API Flask simple. 

L'atelier couvre la chaîne complète de traitement : **Exploration -> Préparation -> Modélisation -> Évaluation -> Déploiement**.

## Structure du Projet
L'arborescence du projet est structurée comme suit :
```
projet_ml_retail/
|-- data/             # Base de données
|   |-- raw/          # Données brutes originales
|   |-- processed/    # Données nettoyées (optionnel)
|   \-- train_test/   # Données splittées (X_train, X_test, etc.)
|-- notebooks/        # Notebooks Jupyter pour l'exploration et le prototypage
|-- src/              # Scripts Python pour la chaîne de production
|   |-- preprocessing.py  # Script de préparation des données et feature engineering
|   |-- train_model.py    # Entraînement du modèle avec GridSearchCV
|   |-- predict.py        # Script de prédiction utilisant le modèle entraîné
|   \-- utils.py          # Fonctions utilitaires (parseurs, imputers, scalers)
|-- models/           # Modèles sauvegardés (scaler.pkl, best_model.pkl, etc.)
|-- app/              # Application web (Flask)
|   \-- app.py        # Point d'entrée de l'API Flask
|-- reports/          # Rapports et visualisations
|-- requirements.txt  # Dépendances du projet (généré via pip freeze)
|-- README.md         # Documentation du projet
\-- .gitignore        # Fichiers à ignorer par git
```

## Instructions d'installation

1. **Création de l'environnement virtuel :**
   ```bash
   python -m venv venv
   ```

2. **Activation de l'environnement :**
   - Sur Windows :
     ```bash
     venv\Scripts\activate
     ```
   - Sur Linux/Mac :
     ```bash
     source venv/bin/activate
     ```

3. **Installation des dépendances :**
   ```bash
   pip install -r requirements.txt
   ```

## Guide d'utilisation

**Étape 1 : Préparation des données**
Placez votre fichier CSV contenant les données brutes dans le répertoire `data/raw/` et renommez-le `dataset.csv`. (S'il n'est pas présent, le script génèrera un jeu de données fictif pour démonstration).
Exécutez le script de preprocessing pour nettoyer les données, créer les nouvelles features, imputer les valeurs manquantes et générer les splits d'entraînement et de test :
```bash
python src/preprocessing.py
```

**Étape 2 : Entraînement du modèle**
Une fois les données préparées, lancez l'entraînement du modèle (RandomForest) avec recherche des hyperparamètres (GridSearchCV). Ce script sauvegardera le meilleur modèle et les noms de features dans le dossier `models/` :
```bash
python src/train_model.py
```

**Étape 3 : Tests de prédiction**
Pour tester si le modèle fonctionne correctement sur des données fictives :
```bash
python src/predict.py
```

**Étape 4 : Lancement de l'API Flask**
Pour déployer le modèle localement et interroger l'API :
```bash
python app/app.py
```
Le serveur démarrera sur `http://localhost:5000/`. Vous pouvez envoyer des requêtes POST sur `http://localhost:5000/predict` pour obtenir la probabilité de churn d'un client.
