from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open(r"C:\Users\HP\OneDrive\Desktop\sindhu_major\final_prediction.pickle", "rb"))
column = pickle.load(open(r"C:\Users\HP\OneDrive\Desktop\sindhu_major\scaler.pickle", 'rb'))

@app.route('/')  # route to display the home page
def home():
    return render_template('index.html')  # rendering the home page

@app.route('/about')
def about():
    return render_template("about.html")



@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == "POST":
        # Extracting features from form data
        H = float(request.form['H'])
        epoch = float(request.form['epoch'])
        e = float(request.form['e'])
        a = float(request.form['a'])
        q = float(request.form['q'])
        om = float(request.form['om'])
        W = float(request.form['W'])
        ad = float(request.form['ad'])
        n = float(request.form['n'])
        per = float(request.form['per'])

        # Creating input array for prediction
        x = np.array([[H, epoch, e, a, q, om, W, ad, n, per]])

        # Performing prediction
        prediction = model.predict(x)
        output = prediction[0]

        # Rendering template with prediction result
        return render_template('predict.html', prediction_text='Asteroid Classification is {}'.format(prediction[0]))

    else:
        return render_template('predict.html')


if __name__ == "__main__":
    app.run(debug=True, port=1212)
