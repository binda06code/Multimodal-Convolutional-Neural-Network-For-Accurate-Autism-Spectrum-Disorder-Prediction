from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

MODEL_PATH = r"C:\Users\TI\Documents\fyp\DataSets\image dataset of autism\ASD arcitectures\ASD arcitectures\Xception\xception-selu.h5"
TXT_PATH = r"C:/Users/TI/Documents/fyp/DataSets/image dataset of autism/classes.txt"
IMG_SIZE = (224, 224)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model_and_classes(model_path, txt_path):
    with open(txt_path, 'r') as f:
        content = f.read()
    classes = content.split(',')
    model = tf.keras.models.load_model(model_path)
    return model, classes

model, classes = load_model_and_classes(MODEL_PATH, TXT_PATH)

def predictor(filepaths, model, classes, img_size, mode):
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

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
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
        
        mode = request.form.get('mode', '')
        if mode not in ['ave', '']:
            return jsonify({'error': 'Invalid mode'}), 400
        
        if mode == 'ave':
            ave_class, ave_p = predictor(filepaths, model, classes, IMG_SIZE, mode)
            return jsonify({'average_class': ave_class, 'average_probability': ave_p})
        else:
            pdf = predictor(filepaths, model, classes, IMG_SIZE, mode)
            return pdf.to_json(orient='records')
    
    return render_template('index.html')
z

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)

