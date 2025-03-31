import numpy as np
import pandas as pd
import pickle
import json

from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score

import logging

logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def read_data_from_prev_step(train_data: str, test_data: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        return pd.read_csv(train_data), pd.read_csv(test_data)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return pd.DataFrame(), pd.DataFrame()
    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty data error: {e}")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        logger.error(f"Error reading data: {e}")
        return pd.DataFrame(), pd.DataFrame()


def model_evaluation(test: pd.DataFrame) -> None:
    try:
        X_test_bow = test.iloc[:, :-1].values
        y_test = test.iloc[:, -1].values

        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)

        # Make predictions
        y_pred = model.predict(X_test_bow)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)

        metrics_dict = {
            'accuracy': accuracy
        }

        with open('metrics.json', 'w') as file:
            json.dump(metrics_dict, file, indent=4)
    
    except FileNotFoundError as e:
        print(f"File Not Found Error: {e}")
    except KeyError as e:
        print(f"Key Error: {e}")
    except ValueError as e:
        print(f"Value Error: {e}")
    except pickle.PickleError as e:
        print(f"Pickle Error: {e}")
    except Exception as e:
        print(f"Unexpected error in model_evaluation function: {e}")

def main():
    try:
        train, test = read_data_from_prev_step('./data/features/train_bow.csv', './data/features/test_bow.csv')
        model_evaluation(test)
    except FileNotFoundError as e:
        print(f"File Not Found Error: {e}")
    except KeyError as e:
        print(f"Key Error: {e}")
    except ValueError as e:
        print(f"Value Error: {e}")
    except Exception as e:
        print(f"Unexpected error in main function: {e}")
        
if __name__ == '__main__':
    main()
