import os
import sys
from geopy.distance import geodesic
from src.logger import logging
from src.exception import CustomException
import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTranformation

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data ingestion methods starts')
        try:
            df = pd.read_csv(os.path.join('notebooks','data','Zomato.csv'))
            logging.info('Dataset read as pandas Dataframe')

            df = df.drop(["ID","Delivery_person_ID","Delivery_person_Age",'Time_Orderd','Time_Order_picked', "Delivery_person_Ratings","Order_Date","Festival"],axis=1)

            df['Distance'] = df.apply(lambda row: round(geodesic((row['Restaurant_latitude'], row['Restaurant_longitude']), (row['Delivery_location_latitude'], row['Delivery_location_longitude'])).km,2), axis=1,)

            condition = df['Distance'] > 20
            df.drop(df[condition].index, inplace=True)
            
            logging.info("EDA Completed performaing ingestion")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info('Train test split')

            train_set, test_set = train_test_split(df,test_size=20)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of data completed")

            return (self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path
                )

        except Exception as e:
            logging.info('Exception occured at data ingestion stage')
            raise CustomException(e, sys)


if __name__=='__main__':
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()      
    data_transformation = DataTranformation()
    train_arr,test_arr,_= data_transformation.initiate_data_tranformation(train_data_path,test_data_path)
