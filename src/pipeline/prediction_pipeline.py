import sys 
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model_path = os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
        
        except Exception as e:
            logging.info('exception occured in prediction')
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                 Distance:float,
                 multiple_deliveries:float,
                 Vehicle_condition:float,
                 Weather_conditions:str,
                 road_traffic:str,
                 Type_of_order:str,
                 type_of_vehicle:str,
                 City:str):
        self.Distance = Distance
        self.multiple_deliveries = multiple_deliveries
        self.Vehicle_condition = Vehicle_condition
        self.Weather_conditions = Weather_conditions
        self.road_traffic = road_traffic
        self.Type_of_order = Type_of_order
        self.type_of_vehicle = type_of_vehicle
        self.City = City

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Distance':[self.Distance],
                'multiple_deliveries':[self.multiple_deliveries],
                'Vehicle_condition':[self.Vehicle_condition],
                'Weather_conditions':[self.Weather_conditions],
                'Road_traffic_density':[self.road_traffic],
                'Type_of_order':[self.Type_of_order],
                'Type_of_vehicle':[self.type_of_vehicle],
                'City':[self.City]
            }
            
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df

        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e, sys)  