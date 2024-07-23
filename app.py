import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the trained model
model = load_model(r'C:\Users\arees\OneDrive\Desktop\LabCVProject\YogaNet_model.h5')

# Dictionary to map predicted class indices to class names
dic = {0: 'downdog', 1: 'goddess', 2: 'plank', 3: 'tree', 4: 'warrior2'}

def predict_label(img_path, model):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))  # Correct target size
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the class
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)  
    return dic[predicted_class]

@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename
        img.save(img_path)
        prediction = predict_label(img_path, model)
        return render_template("index.html", prediction=prediction, img_path=img_path)

if __name__ == "__main__":
    app.run(debug=True)
