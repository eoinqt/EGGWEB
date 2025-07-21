from flask import Flask, render_template, request
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('egg_quality_model.h5')

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

FEATURE_NAMES = [
    "Yolk color", "Egg-white color", "Brightness",
    "Air cells", "Dark Speckles", "Egg Shell Cracks"
]

def preprocess_image(file_path):
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError("Could not load image.")
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(file_path) 

            try:
                img = preprocess_image(file_path)
                prediction = model.predict(img)[0]  #  Sends the image to the AI model gets 6 numbers back, like: [0.62, 0.71, 0.66, 0.50, 0.79, 0.64]
                feature_values = [round(val * 100, 2) for val in prediction]
                avg_quality = round(np.mean(feature_values), 2)     # turns 6 values into percentages and averages them: (62 + 71 + 66 + 50 + 79 + 64) / 6 → 65.33%

                if avg_quality >= 90:
                    verdict = "GOOD quality egg (Safe to consume)"
                elif avg_quality >= 60:
                    verdict = "MEDIUM quality egg (Safe to consume)"
                else:
                    verdict = "BAD quality egg (Not safe to consume)"

                return render_template('index.html',
                    filename=image_file.filename,
                    features=zip(FEATURE_NAMES, feature_values),
                    quality=avg_quality,
                    verdict=verdict,
                    auto_next=True
                )

            except Exception as e:
                return f"Prediction failed: {str(e)}"
    return render_template('index.html', features=[])

if __name__ == '__main__':
    app.run(debug=True)
