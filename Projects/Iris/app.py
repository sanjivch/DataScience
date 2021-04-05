import numpy as np 
from flask import Flask, render_template, request
from sklearn.linear_model import LogisticRegression
import joblib

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])

        X_vals = np.array([sepal_length, sepal_width, petal_length, petal_width])
        X_vals = X_vals.reshape(1,-1)
        lr = joblib.load('iris_lr.pkl')   
        predicted = lr.predict(X_vals)    
        print(predicted[0])

    return render_template('index.html', predicted = predicted[0]) 


if __name__ == '__main__':
    app.run(debug=True)