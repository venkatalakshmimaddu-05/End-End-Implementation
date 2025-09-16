import pickle
from flask import Flask,jsonify,request,render_template
import numpy as np
# import pandas as pd
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application

# import ridge regression and standard scaler pickle
ridge_model=pickle.load(open('models/ridge.pkt','rb'))
scaler_model=pickle.load(open('models/scaler.pkt','rb'))

@app.route("/")
def index():
    return render_template('home.html')


@app.route("/predict", methods=['GET',"POST"])
def predict_datapoint():
    if request.method=="POST":
        Temperature=float(request.form.get("Temperature"))
        RH=float(request.form.get("RH"))
        Ws=float(request.form.get("Ws"))
        Rain=float(request.form.get("Rain"))
        FFMC=float(request.form.get("FFMC"))
        DMC=float(request.form.get("DMC"))
        ISI=float(request.form.get("ISI"))
        Classes=float(request.form.get("Classes"))
        Region=float(request.form.get("Region"))

        columns = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'ISI', 'Classes', 'Region']
        # new_data = pd.DataFrame([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]], columns=columns)
        new_data_scaled = scaler_model.transform(columns)
        result=ridge_model.predict(new_data_scaled)

        return render_template('home.html',results=result[0])
    else:
        return render_template("home.html")

if __name__=="__main__":
    app.run(host="0.0.0.0")