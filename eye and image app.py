from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
import os

#template_dir = r'C:\Users\TI\Documents\GitHub\Deployment-Deep-Learning-Model\TopicListing-1.0.0'
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Paths
IMAGE_MODEL_PATH = r"C:\Users\TI\Documents\fyp\DataSets\image dataset of autism\ASD arcitectures\ASD arcitectures\Xception\xception-selu.h5"
TXT_PATH = r"C:/Users/TI/Documents/fyp/DataSets/image dataset of autism/classes.txt"
KNN_MODEL_PATH = r'C:/Users/TI/Documents/fyp/DataSets/eye tracking dataset/Eye-Tracking Dataset/Eye-Tracking Dataset/knn_model.pkl'
IMG_SIZE = (224, 224)

# Load models and classes
def load_model_and_classes(model_path, txt_path):
    try:
        with open(txt_path, 'r') as f:
            content = f.read()
        classes = content.split(',')
        model = tf.keras.models.load_model(model_path)
        print("Model and classes loaded successfully.")
        return model, classes
    except Exception as e:
        print(f"Error loading model and classes: {e}")
        return None, None

try:
    image_model, classes = load_model_and_classes(IMAGE_MODEL_PATH, TXT_PATH)
    eye_tracking_model = joblib.load(KNN_MODEL_PATH)
    expected_columns = eye_tracking_model.feature_names_in_
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predictor(filepaths, model, classes, img_size, mode):
    try:
        num_classes = len(classes)
        sum_list = [0 for _ in range(num_classes)]
        
        df = pd.DataFrame(filepaths, columns=['filepaths'])
        gen = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_dataframe(df, x_col='filepaths', y_col=None, target_size=img_size, class_mode=None, shuffle=False)
        preds = model.predict(gen)
        
        cclass, pclass = [], []
        for p in preds:
            for j, cp in enumerate(p):
                sum_list[j] += cp
            index = np.argmax(p)
            klass = classes[index]
            cclass.append(klass)
            pclass.append(p[index] * 100)
        
        pdf = pd.DataFrame({'Filename': [os.path.basename(fp) for fp in filepaths], 'Predicted Class': cclass, '% Probability': pclass})
        ave_index = np.argmax(sum_list)
        ave_class = classes[ave_index]
        ave_p = sum_list[ave_index] * 100 / len(filepaths)
        
        if mode == 'ave':
            return ave_class, ave_p
        else:
            return pdf
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

def eye_tracking_predict(file):
    try:
        data = pd.read_csv(file)
        print(f"Data loaded successfully: {data.head()}")

        def preprocess_input(data):
            categorical_columns = ['Stimulus', 'Color', 'Category Group', 'Category Right', 'Category Left', 'Gender']
            data = pd.get_dummies(data, columns=categorical_columns)
            for col in expected_columns:
                if col not in data.columns:
                    data[col] = 0
            data = data[expected_columns]
            return data

        data = preprocess_input(data)
        print(f"Data after preprocessing: {data.head()}")
        missing_cols = [col for col in expected_columns if col not in data.columns]
        if missing_cols:
            return f'Missing columns: {missing_cols}', None

        predictions = eye_tracking_model.predict(data)
        print(f"Predictions: {predictions}")
        results = ['Autistic' if pred == 1 else 'Not Autistic' for pred in predictions]
        output = list(zip(data.index, results))
        print(f"Output: {output}")
        return None, output
    except Exception as e:
        print(f"Error in eye tracking prediction: {e}")
        return str(e), None

# Routes
@app.route('/')
def home():
    return render_template('mainpage.html')

@app.route('/eye_tracking')
def eye_tracking():
    return render_template('home.html')

@app.route('/image_pred')
def image_prediction():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

# Eye tracking prediction
@app.route('/predict_eye_tracking', methods=['POST'])
def predict_eye_tracking():
    if 'file' not in request.files:
        return render_template('home.html', prediction_text='No file uploaded.')

    file = request.files['file']
    error, results = eye_tracking_predict(file)
    if error:
        return render_template('home.html', prediction_text=f"Error: {error}")
    
    return render_template('home.html', prediction_text='Predictions complete. See below.', results=results)

# Image prediction
@app.route('/predict_image', methods=['POST'])
def predict_image():
    if request.method == 'POST':
        try:
            files = request.files.getlist('file')
            if not files:
                return jsonify({'error': 'No files uploaded'}), 400
            
            filepaths = []
            for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    filepaths.append(filepath)
            print(f"Filepaths: {filepaths}")
            
            mode = request.form.get('mode', '')
            if mode not in ['ave', '']:
                return jsonify({'error': 'Invalid mode'}), 400
            
            if mode == 'ave':
                ave_class, ave_p = predictor(filepaths, image_model, classes, IMG_SIZE, mode)
                print(f"Average class: {ave_class}, Average probability: {ave_p}")
                return jsonify({'average_class': ave_class, 'average_probability': ave_p})
            else:
                pdf = predictor(filepaths, image_model, classes, IMG_SIZE, mode)
                print(f"Prediction DataFrame: {pdf}")
                return pdf.to_json(orient='records')
        except Exception as e:
            print(f"Error in image prediction: {e}")
            return jsonify({'error': str(e)}), 500

    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True, host='0.0.0.0', port=5001)
