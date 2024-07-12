from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))
iris_species = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('IRIS.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    features = [np.array(data)]
    
    prediction = model.predict(features)
    predicted_species = iris_species[prediction[0]]
    
    return jsonify({'prediction': predicted_species})
if __name__ == '__main__':
    app.run(debug=True)
