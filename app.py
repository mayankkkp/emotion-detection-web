import cv2
import numpy as np
import gradio as gr
from keras.models import model_from_json

json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def predict_emotion(image):
    if image is None:
        return None, "Please upload an image"
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=7, minSize=(80,80))
    if len(faces) == 0:
        return image, "No face detected"
    results = []
    for (x,y,w,h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48,48)).astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=-1)
        roi = np.expand_dims(roi, axis=0)
        pred = model.predict(roi, verbose=0)
        emotion = labels[np.argmax(pred)]
        conf = float(np.max(pred))
        results.append(f"{emotion} ({conf:.1%})")
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(image, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    return image, "Detected: " + ", ".join(results)

demo = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Image(label="Upload a face image", type="numpy"),
    outputs=[gr.Image(label="Detection result"), gr.Textbox(label="Emotion prediction")],
    title="Facial Emotion Recognition",
    description="Upload an image containing a face; the system detects faces and predicts emotion."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)