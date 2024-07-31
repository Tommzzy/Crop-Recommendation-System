from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

# Load the dataset
file_path = r'C:\Users\USER\Desktop\final_test\Crop_Database.csv'  # Update this path if necessary
crop_data = pd.read_csv(file_path)

# Identify missing values
print("Missing values in each column:")
print(crop_data.isnull().sum())

# Handling missing values
# Impute missing values for numerical columns with mean
numeric_columns = crop_data.select_dtypes(include=['float64', 'int64']).columns
crop_data[numeric_columns] = crop_data[numeric_columns].fillna(crop_data[numeric_columns].mean())

# For the 'label' column, fill missing values with the mode (most frequent value)
crop_data['label'].fillna(crop_data['label'].mode()[0], inplace=True)

# Verify no missing values remain
print("Missing values after imputation:")
print(crop_data.isnull().sum())

# Separate features and target variable
X = crop_data.drop('label', axis=1)
y = crop_data['label']

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Train the model
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_balanced, y_train_balanced)

# Predict on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Create the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    N = float(data['N'])
    P = float(data['P'])
    K = float(data['K'])
    temperature = float(data['temperature'])
    humidity = float(data['humidity'])
    ph = float(data['ph'])
    rainfall = float(data['rainfall'])

    # Create a DataFrame for the input
    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                              columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    
    # Predict the crop probabilities
    crop_probabilities = rf_classifier.predict_proba(input_data)[0]
    
    # Debug statement
    print("Crop probabilities:", crop_probabilities)
    
    # Sort the crops by their predicted probability in descending order
    sorted_indices = crop_probabilities.argsort()[::-1]
    sorted_crops = label_encoder.inverse_transform(sorted_indices)
    sorted_probabilities = crop_probabilities[sorted_indices]
    
    # Define a dictionary mapping crops to image filenames
    crop_images = {
        'rice': 'rice.png',
        'maize': 'maize.png',
        'kidneybeans': 'beans.png',
        'banana': 'bananas.png',
        'mango': 'mango.png',
        'grapes': 'grapes.png',
        'watermelon': 'watermelon.png',
        'apple': 'apple.png',
        'orange': 'orange.png',
        'cotton': 'cotton.png',
        'coffee': 'coffee.png',
        'chickpea': 'chickpea.png',
        'pigeonpeas': 'pigeonpeas.png',
        'mothbeans': 'mothbeans.png',
        'mungbean': 'mungbeans.png',
        'blackgram': 'blackgram.png',
        'lentil': 'lentil.png',
        'pomegranate': 'pomegranate.png',
        'muskmelon': 'muskmelon.png',
        'papaya': 'papaya.png',
        'coconut': 'coconut.png',
        # Add entries for all crops
    }
    
    # Prepare the results with image filenames
    results = [{'crop': crop, 'probability': prob, 'image': crop_images.get(crop, 'default.png')} for crop, prob in zip(sorted_crops, sorted_probabilities) if prob > 0]
    
    # Debug statement
    print("Results:", results)
    
    return jsonify({'predicted_crops': results})



if __name__ == '__main__':
    app.run(debug=True)
