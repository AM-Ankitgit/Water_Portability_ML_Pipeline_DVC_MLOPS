import numpy as np
import pandas as pd

import pickle
import json
from pathlib import Path

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

def load_data(filepath : str) -> pd.DataFrame:
    try:
         return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath}:{e}")
#test_data = pd.read_csv("../data/processed/test_processed.csv")

# X_test = test_data.iloc[:,0:-1].values
# y_test= test_data.iloc[:,-1].values

def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame,pd.Series]:
    try:
        X = data.drop(columns=['Potability'],axis=1)
        y = data['Potability']
        return X,y
    except Exception as e:
        raise Exception(f"Error Preparing data:{e}")


def load_model(filepath:str):
    try:
        with open(filepath,"rb") as file:
            model= pickle.load(file)
        return model
    except Exception as e:
        raise Exception(f"Error loading model from {filepath}:{e}")
    
#model = pickle.load(open("model.pkl","rb"))
def evaluation_model(model, X_test:pd.DataFrame, y_test:pd.Series) -> dict:
    try:
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test,y_pred)
        pre = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        f1score = f1_score(y_test,y_pred)


        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test,y_pred)
        import matplotlib.pyplot as plt
        import seaborn as sns

        save_img = sns.heatmap(cm,annot=True)
        plt.savefig("Confusion metrics.png")


        
        import mlflow
        import dagshub
        import os
        
        # dagshub.init(repo_owner='AM-Ankitgit', repo_name='Water_Portability_ML_Pipeline_DVC_MLOPS', mlflow=True)
        # mlflow.set_experiment("Final_Model_1")
        # mlflow.set_tracking_uri("https://dagshub.com/AM-Ankitgit/Water_Portability_ML_Pipeline_DVC_MLOPS.mlflow")

        dagshub_token = os.getenv("DAGSHUB_TOKEN")
        if not dagshub_token:
            raise EnvironmentError("Token not found")
        
        os.environ['MLFLOW_TRACKING_USERNAME'] = 'AM-Ankitgit'
        os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
        mlflow.set_tracking_uri("https://dagshub.com/AM-Ankitgit/Water_Portability_ML_Pipeline_DVC_MLOPS.mlflow")
        mlflow.set_experiment("Final_Model_2")
        
        with mlflow.start_run(run_name='Testing Data'):
            mlflow.log_metric("acc",acc)
            mlflow.log_metric("pre",pre)
            mlflow.log_metric("recall",recall)
            mlflow.log_metric("f1score",f1score)

            mlflow.log_artifact("Confusion metrics.png")
            mlflow.log_artifact(__file__)
            mlflow.set_tag("author",'Ankit')
            
            

        metrics_dict = {

            'acc':acc,
            'precision':pre,
            'recall' : recall,
            'f1_score': f1score
        }
        return metrics_dict
    except Exception as e:
        raise Exception(f"Error evaluating model : {e}")


def save_metrics(metrics:dict,metrics_path:str) -> None:
    try:
        with open(metrics_path,'w') as file:
            json.dump(metrics,file,indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics to {metrics_path}:{e}")
    
def main():
    try:
        test_data_path = Path("data/processed/test_processed_median.csv")
        # test_data_path = "/data/processed/test_processed_median.csv"
        model_path = "models/model_median.pkl"
        metrics_path = "reports/metrics_median.json"

        # pd.read_csv(test_data)
        test_data = load_data(test_data_path)
        # print(test_data)
        X_test,y_test = prepare_data(test_data)
        model = load_model(model_path)

        
        metrics = evaluation_model(model,X_test,y_test)
        save_metrics(metrics,metrics_path)
    except Exception as e:
        raise Exception(f"An Error occurred:{e}")

if __name__ == "__main__":
    main()