import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer

import yaml

import logging

logger = logging.getLogger('feature_engineering')
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


def missing_value(df : pd.DataFrame) -> None:
    try:
        df.fillna('', inplace=True)
    except AttributeError as e:
        logger.error(f"Error: The input is not a valid DataFrame - {e}")
    except Exception as e:
        logger.error(f"Unexpected error in missing_value function: {e}")

def feature_engineering_train(train : pd.DataFrame, max_features : int) -> tuple[CountVectorizer, pd.DataFrame]:
    try:
        if 'content' not in train.columns or 'final_sentiment' not in train.columns:
            raise KeyError("Required columns ('content', 'final_sentiment') are missing in the train dataset.")
        
        X_train = train['content'].values
        y_train = train['final_sentiment'].values

        vectorizer = CountVectorizer(max_features=max_features)
        X_train_bow = vectorizer.fit_transform(X_train)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train
        return vectorizer, train_df
    
    except KeyError as e:
        logger.error(f"Key Error: {e}")
    except ValueError as e:
        logger.error(f"Value Error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in feature_engineering_train: {e}")

def feature_engineering_test(test : pd.DataFrame, vectorizer : CountVectorizer) -> pd.DataFrame:
    try:
        if 'content' not in test.columns or 'final_sentiment' not in test.columns:
            raise KeyError("Required columns ('content', 'final_sentiment') are missing in the test dataset.")
        
        X_test = test['content'].values
        y_test = test['final_sentiment'].values

        X_test_bow = vectorizer.transform(X_test)

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test
        return test_df
    
    except KeyError as e:
        logger.error(f"Key Error: {e}")
    except ValueError as e:
        logger.error(f"Value Error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in feature_engineering_test: {e}")


# Store the data inside data/features

def save_data(data_folder: str, data_subfolder: str, input_train_data: pd.DataFrame, input_test_data: pd.DataFrame, output_train_data: str, output_test_data: str) -> None:
    try:
        data_path = os.path.join(data_folder, data_subfolder)
        os.makedirs(data_path, exist_ok=True)
        input_train_data.to_csv(os.path.join(data_path, output_train_data), index=False)
        input_test_data.to_csv(os.path.join(data_path, output_test_data), index=False)
        logger.debug(f"Data saved successfully to {data_path}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")



def main():
    try:
        max_features = load_params('params.yaml', 'feature_engineering', 'max_features')
        train, test = read_data_from_prev_step('./data/processed/train_processed.csv', './data/processed/test_processed.csv')
        
        missing_value(train)
        missing_value(test)
        
        vectorizer, train_df = feature_engineering_train(train, max_features)
        test_df = feature_engineering_test(test, vectorizer)
        
        save_data("data", "features", train_df, test_df, "train_bow.csv", "test_bow.csv")
    
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





