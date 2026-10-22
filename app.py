import json
import os
import cv2
import numpy as np
import gradio as gr
from keras.layers import Dense, InputLayer
from keras.models import model_from_json as _original_model_from_json


_original_dense_init = Dense.__init__
def _patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    return _original_dense_init(self, *args, **kwargs)
Dense.__init__ = _patched_dense_init

_original_input_init = InputLayer.__init__
def _patched_input_init(self, *args, **kwargs):
    if 'batch_shape' in kwargs:
        kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
    kwargs.pop('optional', None)
    return _original_input_init(self, *args, **kwargs)
InputLayer.__init__ = _patched_input_init

def model_from_json(json_string, custom_objects=None):
    config = json.loads(json_string)
    
    def clean_config(obj):
        if isinstance(obj, dict):
            # Remove newer Keras parameters
            obj.pop('quantization_config', None)
            obj.pop('batch_shape', None)
            obj.pop('optional', None)

            # Convert newer Keras dtype dictionary to string (e.g., {"class_name": "DTypePolicy", "config": {"name": "float32"}} -> "float32")
            if 'dtype' in obj and isinstance(obj['dtype'], dict):
                dtype_info = obj['dtype']
                if 'config' in dtype_info and 'name' in dtype_info['config']:
                    obj['dtype'] = dtype_info['config']['name']
                elif 'name' in dtype_info:
                    obj['dtype'] = dtype_info['name']
                else:
                    obj['dtype'] = 'float32'
            
            # Fix InputLayer batch_shape
            if 'class_name' in obj and obj['class_name'] == 'InputLayer' and 'config' in obj:
                if 'batch_input_shape' not in obj['config'] and 'batch_shape' in obj['config']:
                    obj['config']['batch_input_shape'] = obj['config']['batch_shape']
            
            for k, v in obj.items():
                clean_config(v)
        elif isinstance(obj, list):
            for item in obj:
                clean_config(item)
                
    clean_config(config)
    return _original_model_from_json(json.dumps(config), custom_objects)

# ==============================================================================
# APPLICATION LOGIC
# ==============================================================================

# Load the model using the patched loader
with open("emotiondetector.json", "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# The label order must exactly match the order used during training
labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def predict_emotion(image):
    if image is None:
        return None, "Please upload an image."
    
    # Convert RGB (from Gradio) to Grayscale (for model)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.2, 
        minNeighbors=7, 
        minSize=(80, 80)
    )
    
    if len(faces) == 0:
        return image, "No face detected. Please upload a clear, front-facing photo."
        
    results = []
    for (x, y, w, h) in faces:
        # Extract face region and preprocess
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float32") / 255.0
        
        # Reshape for the model: (1, 48, 48, 1)
        roi = np.expand_dims(roi, axis=-1)
        roi = np.expand_dims(roi, axis=0)
        
        # Predict
        pred = model.predict(roi, verbose=0)
        emotion = labels[np.argmax(pred)]
        confidence = float(np.max(pred))
        results.append(f"{emotion} ({confidence:.1%})")
        
        # Draw bounding box and label on the image
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
    return image, "Detected: " + ", ".join(results)

# Gradio Web Interface
demo = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Image(label="Upload a face image", type="numpy"),
    outputs=[
        gr.Image(label="Detection Result"),
        gr.Textbox(label="Emotion Prediction")
    ],
    title="Facial Emotion Recognition",
    description="Upload an image containing a face. The system will detect the face and predict the emotion."
)

if __name__ == "__main__":
    # Render dynamically assigns a port via the PORT environment variable
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)