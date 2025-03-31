import pandas as pd
import numpy as np
import os
import nltk
from nltk.stem import SnowballStemmer, WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))
import re
import logging

logger = logging.getLogger('data preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

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

# Data preprocessing

def lower_lemma_stopword_digit(df_copy: pd.DataFrame) -> pd.DataFrame:
    try:
        df_copy['content'] = df_copy['content'].apply(lambda x: x.lower())
        df_copy['content'] = df_copy['content'].apply(lambda x: lemmatizer.lemmatize(x))
        df_copy['content'] = df_copy['content'].apply(lambda x: " ".join([i for i in x.split() if i not in stop_words]))
        df_copy['content'] = df_copy['content'].apply(lambda x: ''.join([i for i in x if not i.isdigit()]))
        return df_copy
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        return df_copy

def removing_punctuations(text: str) -> str:
    try:
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[]^_`{|}~"""), ' ', text)
        text = text.replace('؛', "", )
        text = re.sub('\s+', ' ', text)
        return text.strip()
    except Exception as e:
        logger.error(f"Error removing punctuations: {e}")
        return text

def removing_urls(text: str) -> str:
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except Exception as e:
        logger.error(f"Error removing URLs: {e}")
        return text

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = lower_lemma_stopword_digit(df)
        df['content'] = df['content'].apply(lambda content: removing_punctuations(content))
        df['content'] = df['content'].apply(lambda content: removing_urls(content))
        return df
    except Exception as e:
        logger.error(f"Error normalizing text: {e}")
        return df

# Store the data inside data/processed

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
    train_data, test_data = read_data_from_prev_step('./data/raw/train.csv', './data/raw/test.csv')
    if train_data.empty or test_data.empty:
        logger.error("Train or test data is empty. Exiting.")
        return
    
    train_processed_data = normalize_text(train_data)
    test_processed_data = normalize_text(test_data)
    
    save_data("data", "processed", train_processed_data, test_processed_data, "train_processed.csv", "test_processed.csv")

if __name__ == '__main__':
    main()
