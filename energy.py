from binascii import Incomplete
from flask import Flask, request, render_template
import pickle
import numpy as np
from datetime import datetime
import calendar
    

app = Flask(__name__)
model1 = pickle.load(open('models/model1.pkl', 'rb'))
model2 = pickle.load(open('models/model.pkl', 'rb'))
@app.route('/')
@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/others')
def other_trends():
    return render_template('other_trends.html')

@app.route('/consumptions')
def consumptions():
    return render_template('consumption.html')

@app.route('/tariffs')
def tariffs():
    return render_template('tariffs.html')

@app.route('/clusters')
def get_cluster():
    return render_template('clusters.html')

@app.route('/questions')
def questions():
    return render_template('questions.html')

@app.route('/predict',methods=['POST'])
    
def predict():
    input = request.form
    daily_standing_charge = 50
    print("form input: ",input)
    residents = input["residents"]
    appliances = input["appliances"]
    income_group = input["income"]
    features2 = [[int(appliances),int(residents),int(income_group)]]
    features1 = [[int(appliances),int(residents)]]
    #features = [np.array(input)]  #Convert to the form [[a, b]] for input to the model
    if int(income_group) == 0:
        consumption = model1.predict(features1)
    else:
        consumption = model2.predict(features2) 
    consumption = round(consumption[0],2)
  
    now = datetime.now()
    days = calendar.monthrange(now.year, now.month)[1]
    minCost = round((daily_standing_charge*days + consumption*28)/100,2)
    maxCost = round((daily_standing_charge*days + consumption*30)/100,2)
    results = [consumption, minCost, maxCost]
    return render_template('results.html', results=results)

