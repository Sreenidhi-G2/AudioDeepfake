from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import librosa
from tensorflow import keras
import tempfile
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Load your trained model
MODEL_PATH = r'D:\Projects\Rakshu\backend\model\final_model.keras'
model = None

def load_model():
    global model
    try:
        model = keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_audio(file_path, sr=16000, duration=5, target_length=100):
    """
    Preprocess audio file for the model.
    Model expects shape: (None, 100, 40, 1)
    """
    try:
        # Load audio file
        audio, sample_rate = librosa.load(file_path, sr=sr, duration=duration)
        
        # Extract MFCC features (40 features)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
        # Transpose to get (time_steps, features)
        mfcc = mfcc.T
        
        # Pad or truncate to target_length (100 time steps)
        if mfcc.shape[0] < target_length:
            # Pad with zeros if too short
            pad_width = target_length - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
        else:
            # Truncate if too long
            mfcc = mfcc[:target_length, :]
        
        # Normalize
        mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
        
        # Reshape to (1, 100, 40, 1) - add batch and channel dimensions
        mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension: (1, 100, 40)
        mfcc = np.expand_dims(mfcc, axis=-1)  # Add channel dimension: (1, 100, 40, 1)
        
        return mfcc
    except Exception as e:
        raise Exception(f"Error preprocessing audio: {str(e)}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    
    # Check if model is loaded
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please check server logs.'
        }), 500
    
    # Check if file is present
    if 'audio' not in request.files:
        return jsonify({
            'error': 'No audio file provided'
        }), 400
    
    file = request.files['audio']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({
            'error': 'No file selected'
        }), 400
    
    # Check file extension
    if not allowed_file(file.filename):
        return jsonify({
            'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400
    
    temp_path = None
    try:
        # Save file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        # Preprocess audio
        audio_features = preprocess_audio(temp_path)
        
        # Make prediction
        prediction = model.predict(audio_features)
        
        # Process prediction result
        # Adjust based on your model's output
        confidence = float(prediction[0][0])
        is_fake = confidence > 0.5
        
        # Clean up temporary file
        os.remove(temp_path)
        
        # Return result
        return jsonify({
            'success': True,
            'prediction': {
                'is_deepfake': is_fake,
                'confidence': confidence,
                'label': 'Deepfake' if is_fake else 'Real'
            }
        })
    
    except Exception as e:
        # Clean up on error
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 500
    
    try:
        return jsonify({
            'input_shape': str(model.input_shape),
            'output_shape': str(model.output_shape),
            'total_params': model.count_params()
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'error': 'File too large. Maximum size is 16MB.'
    }), 413

if __name__ == '__main__':
    load_model()
    # Disable reloader to prevent restart issues with librosa/numba
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)