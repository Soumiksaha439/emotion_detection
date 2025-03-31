import xgboost as xgb
import os
import pandas as pd
import pickle
import yaml

import logging

logger = logging.getLogger('model_development')
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


def load_params(params_path: str, stage, parameter: str) -> float:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug("Test size retrieved")
        return params[stage][parameter]
    except FileNotFoundError:
        logger.error(f"Error: The file {params_path} was not found.")
        return None
    except KeyError:
        logger.error(f"Error: Key {stage} or {parameter} not found in parameters file.")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        return None
    




# Import data from previous step

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


def model_training(train: pd.DataFrame, n_estimators: int) -> None:
    try:
        X_train_bow = train.iloc[:, :-1].values
        y_train = train.iloc[:, -1].values

        xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=n_estimators)
        xgb_model.fit(X_train_bow, y_train)

        pickle.dump(xgb_model, open('model.pkl', 'wb'))
    except KeyError as e:
        logger.error(f"Key Error: {e}")
    except ValueError as e:
        logger.error(f"Value Error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in model_training function: {e}")

def main():
    try:
        n_estimators = load_params('params.yaml', 'model_development', 'n_estimators')
        train, test = read_data_from_prev_step('./data/features/train_bow.csv', './data/features/test_bow.csv')
        model_training(train, n_estimators)
    except FileNotFoundError as e:
        logger.error(f"File Not Found Error: {e}")
    except KeyError as e:
        logger.error(f"Key Error: {e}")
    except ValueError as e:
        logger.error(f"Value Error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in main function: {e}")

    

if __name__ == '__main__':
    main()