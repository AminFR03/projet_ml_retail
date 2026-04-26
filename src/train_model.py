import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

def main():
    print("Loading train/test data...")
    train_x_path = 'data/train_test/X_train.csv'
    test_x_path = 'data/train_test/X_test.csv'
    train_y_path = 'data/train_test/y_train.csv'
    test_y_path = 'data/train_test/y_test.csv'

    if not os.path.exists(train_x_path):
        print("Processed data not found. Please run src/preprocessing.py first.")
        return

    X_train = pd.read_csv(train_x_path)
    X_test = pd.read_csv(test_x_path)
    y_train = pd.read_csv(train_y_path).values.ravel()
    y_test = pd.read_csv(test_y_path).values.ravel()

    print("Setting up RandomForest and GridSearchCV...")
    rf = RandomForestClassifier(random_state=42)
    
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
    
    print("Training model...")
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best parameters found: {grid_search.best_params_}")
    
    print("Evaluating model...")
    y_pred = best_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    print("Saving model...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_model.pkl')
    # Save feature names to ensure prediction consistency
    joblib.dump(list(X_train.columns), 'models/feature_names.pkl')
    print("Model saved to models/best_model.pkl")

if __name__ == '__main__':
    main()
