from flask import Flask, request, render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application=Flask(__name__)

app =application

@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')

    else:
        data=CustomData(
            Distance=float(request.form.get('Distance')),
            multiple_deliveries=float(request.form.get('multiple_deliveries')),
            Vehicle_condition=float(request.form.get('Vehicle_condition')),
            Weather_conditions= request.form.get('Weather_conditions'),
            road_traffic = request.form.get('road_traffic'),
            Type_of_order = request.form.get('Type_of_order'),
            type_of_vehicle = request.form.get('type_of_vehicle'),
            City = request.form.get('City')
        )
        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results = round(pred[0],2)

        return render_template('result.html',final_result=results)