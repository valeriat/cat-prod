# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    # Data Preparation
    data = pd.read_csv(input_filepath)
    # Data Exploration
    logger.info(f"Read csv file {input_filepath},\n here are the first rows:\n {data.head().to_string()}")
    logger.info(f"The dataframe shape is {data.shape}")
    logger.info(f"Na count per column:\n{data.isna().sum()}")
    # Data Cleaning
    data = data.drop(columns="total_bedrooms")
    logger.info("Dropped total_bedrooms column")
    # Train-Test Split
    data_train, data_test = train_test_split(data, test_size=0.33, random_state=0)
    logger.info(f"train-test split done. \n"
                f"the train dataset shape is {data_train.shape},"
                f" the test dataset shape is {data_test.shape}")
    # Select X and y values (predictor and outcome)
    X_train = data_train.drop(columns="median_house_value")
    y_train = data_train["median_house_value"]
    X_test = data_test.drop(columns="median_house_value")
    y_test = data_test["median_house_value"]
    X_train.shape, X_test.shape
    write_train_test_data(output_filepath, X_train, X_test, y_train, y_test)
    logger.info("Writing train-test data done.")

def write_train_test_data(output_filepath, X_train, X_test, y_train, y_test):
    logger=logging.getLogger(__name__)
    output_paths = generate_train_test_addresses(output_filepath)
    logger.info(f"Writing train-test data into\n{output_paths}.")
    X_train.to_csv(output_paths['train_feature_path'], index=False)
    X_test.to_csv(output_paths['test_feature_path'], index=False)
    y_train.to_csv(output_paths['train_target_path'], index=False)
    y_test.to_csv(output_paths['test_target_path'], index=False)


def read_train_test_data(output_filepath):
    output_paths = generate_train_test_addresses(output_filepath)
    X_train = pd.read_csv(output_paths['train_feature_path'])
    X_test =pd.read_csv(output_paths['test_feature_path'])
    y_train = pd.read_csv(output_paths['train_target_path'])
    y_test = pd.read_csv(output_paths['test_target_path'])
    return X_train, X_test, y_train, y_test


def generate_train_test_addresses(processed_path):
    if not os.path.exists(processed_path):
        os.mkdir(processed_path)
    output_paths = dict()
    output_paths['train_feature_path'] = processed_path + '/X_train.csv'
    output_paths['test_feature_path'] = processed_path + '/X_test.csv'
    output_paths['train_target_path'] = processed_path + '/y_train.csv'
    output_paths['test_target_path'] = processed_path + '/y_test.csv'
    return output_paths

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    logging_level = os.getenv("LOGGING_LEVEL")
    logging.basicConfig(level=logging_level, format=log_fmt)
    data_path = os.getenv("DATA_PATH")
    processed_path = os.getenv("PROCESSED_PATH")
    main(data_path, processed_path)