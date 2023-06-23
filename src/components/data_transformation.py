import sys 
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object


@dataclass
class DatatransformationConfig:
    preprocessor_obj_file_path= os.path.join('artifacts','preprocessor.pkl')

class DataTranformation:
    def __init__(self):
        self.data_tranformation_config = DatatransformationConfig()

    def get_data_tranformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            categorical_cols = ['Weather_conditions','Road_traffic_density','Type_of_order','Type_of_vehicle','City']
            numerical_cols = ['Vehicle_condition','multiple_deliveries','Distance']

            road_traffic = ['Jam', 'High', 'Medium', 'Low']
            weather_condition = ['Fog','Stormy','Sandstorms','Windy','Cloudy','Sunny']
            Type_of_order = ['Snack', 'Meal', 'Drinks', 'Buffet']
            type_of_vehicle = ['motorcycle', 'scooter', 'electric_scooter', 'bicycle']
            city = ['Metropolitian', 'Urban', 'Semi-Urban']


            logging.info('pipeline initated')
            
            num_pipeline=Pipeline(
            steps=[
            ('imputer',SimpleImputer(strategy='median')),
            ('scaler',StandardScaler())

            ]

            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[weather_condition, road_traffic, Type_of_order, type_of_vehicle, city])),
                ('scaler',StandardScaler())
                ]

            )

            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            logging.info('pipeline completed')
            return preprocessor

        except Exception as e:
            logging.info('Error in data Tranformation')
            raise CustomException(e,sys)

    def initiate_data_tranformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head: \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head: \n{test_df.head().to_string()}')

            logging.info('obtaining preprocessing object')
            preprocessing_obj = self.get_data_tranformation_object()

            target_column_name = 'Time_taken (min)'
            drop_coulmns = [target_column_name]

            input_feature_train_df = train_df.drop(columns=drop_coulmns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_coulmns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("applying preprocessing object on training and testing datasets")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_tranformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info('preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_tranformation_config.preprocessor_obj_file_path
            )
            preprocessing_obj = self.get_data_tranformation_object()
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")
            raise CustomException(e,sys)   


