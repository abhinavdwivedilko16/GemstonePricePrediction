from flask import Flask,request,render_template
from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline

application=Flask(__name__)
app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predict', methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    else:
        data=CustomData(
            carat=float(request.form.get('carat')),
            depth = float(request.form.get('depth')),
            table = float(request.form.get('table')),
            x = float(request.form.get('x')),
            y = float(request.form.get('y')),
            z = float(request.form.get('z')),
            cut = request.form.get('cut'),
            color= request.form.get('color'),
            clarity = request.form.get('clarity')
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2) #rounding off the result upto 2 decimal plaes

        return render_template('results.html',final_result=results)

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000)