from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_session import Session
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 

from io import BytesIO
from base64 import b64encode
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

app = Flask(__name__)


app.config['SECRET_KEY'] = ' '  
app.config['SESSION_TYPE'] = 'filesystem'  
app.config['SESSION_FILE_DIR'] = os.path.join(os.getcwd(), 'flask_session')
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)
Session(app)


net = cv2.dnn.readNetFromCaffe('DATA/haarcascades/deploy.prototxt.txt',
                               'DATA/haarcascades/res10_300x300_ssd_iter_140000.caffemodel')


deepfake_model = load_model('face.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']
    video_path = os.path.join('uploads', video_file.filename)
    video_file.save(video_path)

    cap = cv2.VideoCapture(video_path)
    deepfake_count = 0
    total_frames = 0
    confidence_scores = []
    frame_histograms = []
    deepfake_frame = None 

    THRESHOLD = 0.7  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")

                
                face_roi = frame[startY:endY, startX:endX]

                
                face_roi_resized = cv2.resize(face_roi, (256, 256))
                face_roi_array = img_to_array(face_roi_resized)
                face_roi_array = np.expand_dims(face_roi_array, axis=0)
                face_roi_array = preprocess_input(face_roi_array)

                
                prediction = deepfake_model.predict(face_roi_array)
                confidence_score = prediction[0][0]
                confidence_scores.append(confidence_score)

                hist_b = cv2.calcHist([frame], [0], None, [256], [0, 256])
                hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
                hist_r = cv2.calcHist([frame], [2], None, [256], [0, 256])
                frame_histograms.append((hist_b.flatten(), hist_g.flatten(), hist_r.flatten()))

                if confidence_score > THRESHOLD:
                    deepfake_count += 1
                    deepfake_frame = frame.copy()  

                    
                    cv2.rectangle(deepfake_frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

        total_frames += 1

    cap.release()


    mean_confidence = np.mean(confidence_scores)
    variance_confidence = np.var(confidence_scores)


    deepfake_detected = mean_confidence > 0.7 and variance_confidence < 0.05

 
    session['analysis_data'] = {
        'deepfake_detected': deepfake_detected,
        'confidence_scores': confidence_scores,
        'frame_histograms': frame_histograms,
        'deepfake_frame': deepfake_frame  
    }

    return redirect(url_for('dashboard', video_name=video_file.filename))

@app.route('/dashboard')
def dashboard():
    video_name = request.args.get('video_name')
    analysis_data = session.get('analysis_data')  

    if analysis_data is None:
        return jsonify({'error': 'Analysis data not found'}), 404

    histograms = analysis_data['frame_histograms']
    confidence_scores = analysis_data['confidence_scores']
    deepfake_detected = analysis_data['deepfake_detected']
    deepfake_frame = analysis_data['deepfake_frame']

  
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

 
    ax[0].plot(confidence_scores, color='blue')
    ax[0].set_title('Confidence Scores Over Frames')
    ax[0].set_xlabel('Frame')
    ax[0].set_ylabel('Confidence Score')

    
    ax[1].plot(histograms[0][0], color='blue', label='Blue')
    ax[1].plot(histograms[0][1], color='green', label='Green')
    ax[1].plot(histograms[0][2], color='red', label='Red')
    ax[1].set_xlabel('Pixel Value')
    ax[1].set_ylabel('Frequency')
    ax[1].legend()

    
    png_image = BytesIO()
    plt.savefig(png_image, format='png')
    png_image.seek(0)
    png_image_base64 = b64encode(png_image.getvalue()).decode('ascii')

   
    if deepfake_frame is not None:
        _, deepfake_frame_png = cv2.imencode('.png', deepfake_frame)
        deepfake_frame_base64 = b64encode(deepfake_frame_png).decode('ascii')
    else:
        deepfake_frame_base64 = None

    return render_template('dashboard.html', image_data=png_image_base64,
                           deepfake_frame_data=deepfake_frame_base64,
                           result='Deepfake Detected' if deepfake_detected else 'Real Video',
                           confidence_scores=confidence_scores,
                           histograms=histograms)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
