# Rapport du Projet: Analyse Comportementale Clientèle Retail

Ce document détaille toutes les étapes qui ont été accomplies afin de répondre aux exigences formulées dans le document `projet.pdf`. La consigne demandait de mettre en place l'environnement et l'architecture d'un projet d'analyse comportementale de clientèle pour une entreprise d'e-commerce, en couvrant l'exploration, la préparation, la modélisation, l'évaluation et le déploiement.

## 1. Mise en Place de l'Arborescence et Environnement

La première exigence était de structurer correctement le projet pour séparer les données, le code source, les modèles, et l'application Web.
- **Dossiers `data/raw/`, `data/processed/`, et `data/train_test/`** : Ces dossiers permettent d'accueillir et de versionner correctement le jeu de données fictif (ou réel lorsqu'il sera introduit) selon son état (brut, nettoyé ou splitté pour l'entraînement).
- **Dossier `src/`** : L'épine dorsale des scripts de production Python, incluant :
  - `utils.py` : Fonctions mutualisées (gestion de la date, nettoyage, feature engineering).
  - `preprocessing.py` : Script automatisé d'ingestion et de préparation de bout en bout.
  - `train_model.py` : Script pour l'entraînement et l'optimisation des modèles.
  - `predict.py` : Script pour tester des prédictions simples.
- **Dossier `app/`** : Contient l'application Flask déployant le modèle.
- **Dossiers `models/` et `reports/`** : Prêts à recevoir les modèles sérialisés (ex. `.pkl`) et les bilans d'entraînement.

## 2. Extraction et Analyse des Besoins (Lecture du PDF)

Pour mener à bien le projet, j'ai commencé par lire le contenu de `projet.pdf` en installant `pypdf` dans l'environnement virtuel. Les informations extraites ont indiqué que la base de données attendue comportait jusqu'à 52 features. Le cas d'étude s'axe principalement autour de la prédiction de "Churn" (le taux de départ des clients).

*Note : N'ayant pas accès au dataset réel initial (`dataset.csv` aux 52 features n'était pas présent dans les dossiers), j'ai configuré les scripts `preprocessing.py` pour générer un dataset factice (dummy) s'il n'en trouve pas, afin de s'assurer que le code est entièrement testable et fonctionnel directement.*

## 3. Développement du Module `src/utils.py`

J'ai centralisé toutes les fonctions clés de manipulation de données recommandées dans la documentation :
- **Parsing** : Traitement des dates (`pd.to_datetime`) pour normaliser les formats comme `12/03/10` et extraction de `RegYear`, `RegMonth`, `RegDay`, et `RegWeekday`.
- **Feature Engineering** : Création de features complexes avec ratios : `MonetaryPerDay`, `AvgBasketValue` (basé sur la fréquence et le total dépensé), et le `TenureRatio`.
- **Nettoyage** : Suppression des features inutiles (ex. colonnes dont la variance est nulle comme `NewsletterSubscribed`).
- **Imputation** : Substitution des valeurs manquantes à l'aide de stratégies `SimpleImputer` sur les features continues et catégorielles.
- **Mise à l'échelle (StandardScaler) & ACP (PCA)** : Normalisation des données pour les algorithmes et structure fonctionnelle pour la réduction de dimension si nécessaire.

## 4. Pipeline de Préparation des Données (`src/preprocessing.py`)

Le script a été rédigé de façon déclarative. Voici la séquence d'exécution :
1. **Chargement** : Récupère les données depuis `data/raw/dataset.csv`.
2. **Transformations de base** : Parse des dates, création de l'ingénierie des caractéristiques et suppression des variables inutiles.
3. **Encodage (One-Hot)** : Transformation des variables catégorielles avec `pd.get_dummies()` avant de séparer le jeu de données pour éviter des incohérences de dimensions.
4. **Train / Test Split** : Découpe en un jeu d'entraînement (80%) et un jeu de test (20%) avec stratification sur la variable cible (`Churn`).
5. **Sauvegardes** : Sauvegarde le `StandardScaler` dans le dossier `models/` pour s'assurer que les futures données seront scalées identiquement, et sauvegarde les datasets splittés dans `data/train_test/`.

## 5. Modélisation et Optimisation (`src/train_model.py`)

J'ai implémenté l'entraînement sur un algorithme de type **Random Forest** (souvent recommandé pour les données tabulaires et de classification de churn) :
- Le script charge les ensembles d'entraînement/test pré-calculés.
- Il exécute une recherche d'hyperparamètres avancée avec **GridSearchCV** pour trouver les paramètres optimaux pour le RandomForest (`n_estimators`, `max_depth`, `min_samples_split`).
- Une fois le meilleur modèle identifié, il est testé vis-à-vis des données réelles (données de test) pour produire un **Rapport de Classification** ainsi que le score de Précision.
- Enfin, ce modèle optimisé et les noms des features utilisées sont stockés dans `models/best_model.pkl` et `models/feature_names.pkl`.

## 6. Déploiement : L'Application Web avec Flask (`app/app.py`)

J'ai rédigé un serveur API REST `app.py` utilisant le framework `Flask` pour déployer le modèle entraîné, tel qu'exigé dans le PDF :
- **Point de terminaison GET (`/`)** : Sert une page d'accueil HTML conviviale expliquant le fonctionnement de l'API et donnant un exemple de structure JSON valide.
- **Point de terminaison POST (`/predict`)** : Accepte un jeu de données JSON formaté, charge le modèle ainsi que le *scaler* en mémoire via une interface fournie dans `predict.py`, retraite les données entrantes dynamiquement, et renvoie une réponse JSON affichant la prédiction de churn et les probabilités.

## 7. Configuration de l'Environnement et de la Documentation

1. **Dépendances** : J'ai installé les paquets nécessaires (`scikit-learn`, `pandas`, `flask`, etc.) via l'environnement virtuel existant, et re-généré le fichier `requirements.txt` à partir des paquets exacts (à l'aide de la commande `pip freeze > requirements.txt`).
2. **README.md** : Le fichier d'explication de base a été redéfini avec le Titre du Projet, le Guide d'Installation complet, l'explication de l'arborescence imposée et un Guide d'Utilisation étape par étape.

**Prochaines Étapes pour Vous :**
- Insérer le vrai fichier csv comportant les 52 features sous le nom `dataset.csv` dans le dossier `data/raw/`.
- Lancer le flux d'exécution dans ce sens : `src/preprocessing.py` > `src/train_model.py` > `app/app.py`.
