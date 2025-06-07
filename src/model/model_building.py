import pandas as pd
import yaml
import pickle
from sklearn.ensemble import RandomForestClassifier

def load_params(params_path: str) -> int:
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        return params["model_building"]["n_estimators"]
    except Exception as e:
        raise Exception(f"Error loading parameters from {params_path}: {e}")

def load_data(data_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(data_path)
    except Exception as e:
        raise Exception(f"Error loading data from {data_path}: {e}")

def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X = data.drop(columns=['Potability'], axis=1)
        y = data['Potability']
        return X, y
    except Exception as e:
        raise Exception(f"Error preparing data: {e}")

def train_model(X: pd.DataFrame, y: pd.Series, n_estimators: int) -> RandomForestClassifier:
    try:

        from sklearn.model_selection import RandomizedSearchCV
        hyperameters = {'n_estimators':[10,20,30,40,60],
                        'max_depth':[None,1,2,3,4,5,6,7,8]}
        
        clf = RandomForestClassifier()
        random_search = RandomizedSearchCV(estimator=clf,param_distributions=hyperameters,cv=3)
        random_search.fit(X, y)

        best_random_clf = random_search.best_estimator_

        best_random_clf.fit(X, y)

        return best_random_clf
    except Exception as e:
        raise Exception(f"Error training model: {e}")

def save_model(model: RandomForestClassifier, model_name: str) -> None:
    try:
        with open(model_name, "wb") as file:
            pickle.dump(model, file)
    except Exception as e:
        raise Exception(f"Error saving model to {model_name}: {e}")

def main():
    try:
        params_path = "params.yaml"
        data_path = "./data/processed/train_processed_median.csv"
        model_name = "models/model_median.pkl"

        n_estimators = load_params(params_path)
        train_data = load_data(data_path)
        X_train, y_train = prepare_data(train_data)

        from sklearn.model_selection import RandomizedSearchCV
        hyperameters = {'n_estimators':[10,20,30,40,60],
                        'max_depth':[None,1,2,3,4,5,6,7,8]}
        
        clf = RandomForestClassifier()

        random_search = RandomizedSearchCV(estimator=clf,param_distributions=hyperameters,cv=3)
        
        import mlflow
        import dagshub
        import os

        
        dagshub.init(repo_owner='AM-Ankitgit', repo_name='Water_Portability_ML_Pipeline_DVC_MLOPS', mlflow=True)
        mlflow.set_experiment("RandomFor")
        mlflow.set_tracking_uri("https://dagshub.com/AM-Ankitgit/Water_Portability_ML_Pipeline_DVC_MLOPS.mlflow")
        
        mlflow.autolog()
        with mlflow.start_run(run_name="Track result of all combination") as parents:
            random_search.fit(X_train, y_train)

            for i in range(len(random_search.cv_results_['params'])):
                mlflow.autolog()
                with mlflow.start_run(run_name=f"Combination{i+1}",nested=True) as child:
                    mlflow.log_params(random_search.cv_results_['params'][i])
                    mlflow.log_metrics({
                        "mean_test_score": random_search.cv_results_['mean_test_score'][i]
                    })
        
            
            best_random_clf = random_search.best_estimator_

            best_random_clf.fit(X_train, y_train)
            save_model(best_random_clf, model_name)
            print("Model trained and saved successfully!")
    except Exception as e:
        
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()