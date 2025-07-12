from flask import Flask, render_template, request
import pickle
import google.generativeai as genai
import keras
from PIL import Image
import numpy as np
import joblib

app = Flask(__name__)

# Load Models
crop_model = joblib.load('Notebook/models/farmmitra_crop_model.pkl')
disease_model = keras.models.load_model('plant_disease_model.h5')

# ✅ Disease class labels
class_labels = [
    'Potato___healthy',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_healthy',
    'Tomato__Target_Spot',
    'Tomato_Septoria_leaf_spot',
    'Pepper__bell___Bacterial_spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato_Bacterial_spot',
    'Pepper__bell___healthy',
    'Tomato_Early_blight',
    'Potato___Late_blight',
    'Potato___Early_blight'
]

# Gemini
genai.configure(api_key="AIzaSyA_eprNsgB-RtxuGMhIlsbwJnHdMHhyRME")
chat_model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/crop', methods=['GET', 'POST'])
def crop_predict():
    if request.method == 'POST':
        nitrogen = float(request.form['nitrogen'])
        phosphorus = float(request.form['phosphorus'])
        potassium = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        data = [[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]]
        prediction = crop_model.predict(data)[0]
        return render_template('crop.html', prediction=prediction)
    return render_template('crop.html')

@app.route('/disease', methods=['GET', 'POST'])
def disease_detect():
    if request.method == 'POST':
        file = request.files['image']
        img = Image.open(file).resize((128, 128))  # ✅ Correct size
        img = img.convert('RGB')
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 128, 128, 3)

        result = disease_model.predict(img_array)
        disease_class_index = np.argmax(result)
        disease_class_name = class_labels[disease_class_index]

        return render_template('disease.html', result=f"Disease: {disease_class_name}")
    return render_template('disease.html')

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        user_input = request.form['question']
        response = chat_model.generate_content(user_input)
        return render_template('chatbot.html', reply=response.text)
    return render_template('chatbot.html')

if __name__ == '__main__':
    app.run(debug=True)
