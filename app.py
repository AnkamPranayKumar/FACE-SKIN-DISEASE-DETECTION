# from flask import Flask, request, jsonify, render_template
# from tensorflow.keras.models import load_model
# import cv2
# import numpy as np

# app = Flask(__name__)
# model = load_model('C:\VS_Project\my_models1.h5')

# def preprocess_input(x):
#     # Preprocessing function to match the model's input preprocessing.
#     # This function should be updated according to the model's documentation.
#     return x / 255.0

# def predict_image(model, image):
#     img = cv2.resize(image, (224,224))
#     img = preprocess_input(np.array([img]))
#     predictions = model.predict(img)
#     return predictions
# @app.route('/', methods=['GET'])
# def home():
#     return render_template('index.html')
# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.files['file'].read()
#     npimg = np.fromstring(data, np.uint8)
#     img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
#     predictions = predict_image(model, img)
#     # Assuming predictions are probabilities for each class, get the max probability
#     max_confidence = np.max(predictions)
#     return jsonify({"confidence": max_confidence.tolist()})

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2
import numpy as np

app = Flask(__name__)
model = load_model('C:\VS_Project\my_models1.h5')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

class_names = ['Acne_cystic', 'Actinic_keratosis', 'Atopic_Dermatitis', 'basal_cell_carcinoma_nose', 'lupus_chronic_cutaneous', 'Rosacea', 'Rosacea_nose', 'Seborrheic_Keratoses', 'Squamous_cell_carcinoma', 'Vascular_lesion']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' in request.files:
        # Predict from uploaded image
        data = request.files['file'].read()
        npimg = np.fromstring(data, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    else:
        # Predict from image file
        image_path = 'path_to_your_image.jpg'
        img = cv2.imread(image_path)
    
    img = cv2.resize(img, (224,224))
    img = preprocess_input(np.array([img]))  # Model expects input as a batch of images

    predictions = model.predict(img)
    predicted_class = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class]
    max_confidence = predictions[0][predicted_class]

    return {"class": predicted_class_name, "confidence": max_confidence.tolist()}


if __name__ == '__main__':
    app.run(debug=True)