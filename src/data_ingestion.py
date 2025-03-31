import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os 
import yaml
import logging

logger = logging.getLogger('data_ingestion')
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

def load_data(data_url: str) -> pd.DataFrame:
    try:
        logger.debug("Data is getting loaded to dataframe")
        return pd.read_csv(data_url)
    except Exception as e:
        logger.error(f"Error loading data from {data_url}: {e}")
        return pd.DataFrame()

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if 'sentiment' not in df.columns:
            raise ValueError("Error: Column 'sentiment' not found in the dataset.")
        df['final_sentiment'] = df['sentiment'].apply(lambda x: 2 if x in ['happiness','love','fun','relief','enthusiasm','surprise']
                                                    else 1 if x == 'neutral' else 0)
        df.drop(['sentiment', 'tweet_id'], axis=1, inplace=True, errors='ignore')
        return df
    except Exception as e:
        logger.error(f"Error in preprocessing data: {e}")
        return df

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
    test_size = load_params('params.yaml', 'data_ingestion', 'test_size')
    if test_size is None:
        logger.error("Error: Unable to load test_size parameter.")
        return
    
    df = load_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
    if df.empty:
        logger.error("Error: Dataframe is empty. Exiting.")
        return
    
    df = preprocess_data(df)
    train, test = train_test_split(df, test_size=test_size, random_state=55)
    save_data("data", "raw", train, test, "train.csv", "test.csv")
    
if __name__ == '__main__':
    main()
