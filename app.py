from flask import Flask, render_template, request
import pickle
import numpy as np

# Load model
with open('model/house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        size = int(request.form['size'])
        location = int(request.form['location'])  # Assuming encoded location

        features = np.array([[bedrooms, bathrooms, size, location]])
        prediction = model.predict(features)[0]
        price_in_inr = prediction * 83
        
        data = {
            'Dollar_Price':round(prediction,2),
            'Rupees_Price': round(price_in_inr,2)
        }

        return render_template('result.html', **data)

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
