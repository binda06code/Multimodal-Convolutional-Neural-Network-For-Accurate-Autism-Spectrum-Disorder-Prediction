from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model using joblib
model_path = r'C:/Users/TI/Documents/fyp/DataSets/eye tracking dataset/Eye-Tracking Dataset/Eye-Tracking Dataset/knn_model.pkl'
model = joblib.load(model_path)

# Check the type of the loaded model
print(f"Loaded model type: {type(model)}")

# Get the expected columns from the model
expected_columns = model.feature_names_in_

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if a file is provided
        if 'file' not in request.files:
            return render_template('home.html', prediction_text='No file uploaded.')

        file = request.files['file']

        # Read the file into a DataFrame
        data = pd.read_csv(file)

        # Preprocess the input data
        def preprocess_input(data):
            # One-hot encode the categorical columns
            categorical_columns = ['Stimulus', 'Color', 'Category Group', 'Category Right', 'Category Left', 'Gender']
            data = pd.get_dummies(data, columns=categorical_columns)
            
            # Ensure all expected columns are present
            for col in expected_columns:
                if col not in data.columns:
                    data[col] = 0  # Add missing column with default value 0

            # Reorder columns to match the training data
            data = data[expected_columns]
            
            return data

        # Preprocess the input data
        data = preprocess_input(data)

        # Validate the input data
        missing_cols = [col for col in expected_columns if col not in data.columns]
        if missing_cols:
            return render_template('home.html', prediction_text=f'Missing columns: {missing_cols}')

        # Make predictions
        predictions = model.predict(data)

        # Convert predictions to human-readable form
        results = ['Autistic' if pred == 1 else 'Not Autistic' for pred in predictions]

        # Prepare output
        output = list(zip(data.index, results))

        # Print output to debug
        print(output)
    except Exception as e:
        return render_template('home.html', prediction_text=f"Error: {str(e)}")

    return render_template('home.html', prediction_text='Predictions complete. See below.', results=output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

'''from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model using joblib
model_path = r'C:/Users/TI/Documents/fyp/DataSets/eye tracking dataset/Eye-Tracking Dataset/Eye-Tracking Dataset/knn_model.pkl'
model = joblib.load(model_path)

# Check the type of the loaded model
print(f"Loaded model type: {type(model)}")



# Inspect the feature names
if hasattr(model, 'feature_names_in_'):
    print("Feature names used during model training:")
    print(model.feature_names_in_)
else:
    print("The model does not have 'feature_names_in_' attribute. It might not have been trained with a DataFrame with named columns.")


# Get the expected columns from the model
expected_columns = model.feature_names_in_


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if a file is provided
        if 'file' not in request.files:
            return render_template('home.html', prediction_text='No file uploaded.')

        file = request.files['file']

        # Read the file into a DataFrame
        data = pd.read_csv(file)

        # Extract the relevant columns
        data = data[expected_columns]
            data = pd.get_dummies(data, columns=categorical_columns)


        # Validate the input data
        if not all(column in data.columns for column in expected_columns):
            return render_template('home.html', prediction_text='Invalid file format. Please upload a CSV with the correct columns.')

        # Make predictions
        predictions = model.predict(data)

        # Convert predictions to human-readable form
        results = ['Autistic' if pred == 1 else 'Not Autistic' for pred in predictions]

        # Prepare output
        output = list(zip(data.index, results))

        # Print output to debug
        print(output)
    except Exception as e:
        return render_template('home.html', prediction_text=f"Error: {str(e)}")

    return render_template('home.html', prediction_text='Predictions complete. See below.', results=output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
    
    
    
    
    
    
    
    
    
    
    
    
    
    '''