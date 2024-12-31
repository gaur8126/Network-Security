import os 
import sys 
import pandas as pd 
import numpy as np 


"""
defining common constant variable for training pipeline
"""

TARGET_COLUMN = "Result"
PIPELINE_NAME:str= "NetworkSecurity"
ARTIFACT_DIR : str = "Artifacts"
FILE_NAME :str = "NetworkData.csv"

TRAIN_FILE_NAME :str = 'train.csv'
TEST_FILE_NAME :str = 'test.csv'

PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"
MODEL_FILE_NAME = "model.pkl"
SCHEMA_FILE_PATH = os.path.join("data_schema","schema.yaml")
SCHEMA_DROP_COLS = "drop_columns"

SAVE_MODEL_DIR = os.path.join("saved_models")

"""
Data Ingestion related constant start with DATA_INGESTION VAR name
"""

DATA_INGESTION_COLLECTION_NAME:str = "NetworkData"
DATA_INGESTION_DATABASE_NAME :str = "NetworkSecurity"
DATA_INGESTION_DIR_NAME :str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR :str = "feature_store"
DATA_INGESTION_INGESTED_DIR :str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION:str = 0.2

"""
Data Validation related constant start with DATA_VALIDATION VAR name
"""


DATA_VALIDATION_DIR_NAME:str   = "data_validation"
DATA_VALIDATION_VALID_DIR:str = "validated"
DATA_VALIDATION_INVALID_DIR:str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR:str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME:str = "report.yaml"

"""
Data Transfromation related constant start with DATA_TRANSFRAMTION VAR name
"""


DATA_TRANSFORMATION_DIR_NAME:str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR :str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR :str = "transformed_object"

DATA_TRANSFORMATION_IMPUTER_PARAMS :str = {
    "missing_values":np.nan,
    "n_neighbors":3,
    "weights":"uniform",
}

DATA_TRANSFORMATION_TRAIN_FILE_PATH :str = "train.npy"
DATA_TRANSFORMATION_TEST_FILE_PATH :str = "test.npy"



"""
Model Trainer   related constant start with MODEL_TRAINER VAR name
"""

