import pickle

#importing the necessary files
from flask import Flask, render_template, jsonify, request
import numpy as np
model = pickle.load(open('final_model.pickle','rb'))

#initialising the flask app
app = Flask(__name__)

@app.route('/')
def home_page():  # To render homepage
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    if request.method == 'POST':
        CRIM = float(request.form['Crime Rate'])
        ZN = float(request.form['Zone'])
        INDUS = float(request.form['Industries'])
        CHAS = int(request.form['Charles River'])
        NOX = float(request.form['Nitric Oxide Conc.'])
        RM = float(request.form['Rooms'])
        AGE = float(request.form['Age'])
        DIS = float(request.form['Distance'])
        RAD = float(request.form['Highways'])
        TAX = float(request.form['Tax'])
        PTRATIO = float(request.form['Teacher Ratio'])
        B = float(request.form['African-American'])
        LSTAT = float(request.form['Lower Status'])


        data = np.array([[CRIM,ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX,PTRATIO, B, LSTAT]])
        mean = np.load('mean.npy')
        std = np.load('std.npy')
        
        #creating a list to apply the Scaling using the mean and Standard Deviation we learned from our training model
        new = [[(data[0][i] - mean[i])/std[i] for i in range(0,13)]]

        my_prediction = model.predict(new)


        return render_template('result.html', prediction = int(my_prediction))

if __name__ == '__main__':
    app.run(debug=True)
