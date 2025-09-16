import numpy as np
from flask import Flask, request, render_template
import pickle

flask_app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@flask_app.route('/')
def home():
    return render_template('webpage.html')
@flask_app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    output = prediction
    return render_template('webpage.html', prediction=output)
# '{}'.format(output))

if __name__ == "__main__":
    flask_app.run(debug=True)