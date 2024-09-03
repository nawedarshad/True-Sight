from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_session import Session
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
from Crypto.PublicKey import RSA
from Crypto.Signature import pkcs1_15
from Crypto.Hash import SHA256
import base64
import shutil
import random

from io import BytesIO
from base64 import b64encode
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

app = Flask(__name__)
key = RSA.generate(2048)
private_key = key.export_key()
public_key = key.publickey().export_key()
# Configuration for Flask session
app.config['SECRET_KEY'] = 'your_secret_key'  # Replace with a strong secret key
app.config['SESSION_TYPE'] = 'filesystem'  # Store session data on the server-side filesystem
app.config['SESSION_FILE_DIR'] = os.path.join(os.getcwd(), 'flask_session')
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)
Session(app)
fake_image_reasons = [
    "Inconsistent lighting: The lighting or shadows on different parts of the image do not match, suggesting possible manipulation. Natural lighting sources cast shadows and illuminate objects consistently. Mismatched lighting or shadows can indicate tampering, such as adding or adjusting elements.",
    
    "Irregular facial features: Facial features may appear distorted or unnatural compared to the rest of the image. Issues like unusual proportions, asymmetry, or unnatural textures can signal that the image has been altered. For example, eyes or mouths might appear disproportionately sized or oddly shaped.",
    
    "Unnatural textures: The texture of the skin or objects in the image might look inconsistent or overly smooth. Manipulated images often exhibit unrealistic textures due to digital effects or editing tools, resulting in surfaces that do not match the natural appearance.",
    
    "Artifacts and distortions: Presence of unusual artifacts, blurriness, or distortion, especially around the edges of objects or faces. Editing processes can introduce artifacts like noise, compression marks, or irregularities that become apparent upon close inspection, such as blurry or uneven edges.",
    
    "Anomalies in reflections: Reflections or shadows do not align correctly with the objects or faces in the image. If reflections appear out of place or mismatch the object they represent, it could indicate that parts of the image have been altered or combined without proper attention to detail.",
    
    "Misaligned features: Features such as eyes, mouth, or ears may be misaligned or asymmetrical in a way that looks unnatural. Facial features should align with underlying bone structure and muscle movement. Misalignments can occur due to digital alterations or blending errors.",
    
    "Image compression issues: Evidence of excessive or unusual image compression artifacts that are not typical for the given context. Heavy editing or saving in lower quality formats can introduce artifacts like blurring or color banding, which may be inconsistent with the image's intended context.",
    
    "Source verification issues: The image source is not trustworthy or the origin of the image cannot be confirmed. Verification issues include a lack of metadata, unverifiable publication history, or suspicious URLs, which can indicate that the image might be manipulated or fabricated.",
    
    "Metadata inconsistencies: Metadata (such as timestamps or camera settings) that does not match the context of the image. Inconsistencies in metadata, such as mismatched creation dates or camera settings, can suggest that the image has been altered or edited.",
    
    "Deepfake detection model results: If a deepfake detection model or algorithm identifies the image as a potential fake. Advanced models analyze images for signs of manipulation, such as unusual facial movements or artifacts. A flag from such models can provide additional evidence of manipulation."
]


# Load the pre-trained face detection model
net = cv2.dnn.readNetFromCaffe('DATA/haarcascades/deploy.prototxt.txt',
                               'DATA/haarcascades/res10_300x300_ssd_iter_140000.caffemodel')

# Load the pre-trained deepfake detection model
deepfake_model = load_model('face.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/input-file.html')
def inputfile():
    return render_template('input-file.html')
@app.route('/api.html')
def api():
    return render_template('api.html')
@app.route('/awarness.html')
def awarness():
    return render_template('awarness.html')

@app.route('/doc.html')
def doc():
    return render_template('doc.html')

@app.route('/game.html')
def game():
    return render_template('game.html')
@app.route('/moderation.html')
def moderation():
    return render_template('moderation.html')

@app.route('/signature.html')
def signature():
    return render_template('signature.html')

@app.route('/generate_signature', methods=['POST'])
def generate_signature():
    data = request.json.get('data')
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    # Generate the signature
    h = SHA256.new(data.encode())
    signature = pkcs1_15.new(RSA.import_key(private_key)).sign(h)
    return jsonify({'signature': base64.b64encode(signature).decode()})

@app.route('/verify_signature', methods=['POST'])
def verify_signature():
    data = request.json.get('data')
    signature = request.json.get('signature')

    if not data or not signature:
        return jsonify({'error': 'Data or signature missing'}), 400

    try:
        # Verify the signature
        h = SHA256.new(data.encode())
        pkcs1_15.new(RSA.import_key(public_key)).verify(h, base64.b64decode(signature))
        return jsonify({'status': 'Signature is valid'})
    except (ValueError, TypeError):
        return jsonify({'status': 'Signature is invalid'})

@app.route('/upload', methods=['POST'])
def upload_media():
    if 'media' not in request.files:
        return jsonify({'error': 'No media file provided'}), 400

    media_file = request.files['media']
    media_path = os.path.join('uploads', media_file.filename)
    media_file.save(media_path)

    file_extension = os.path.splitext(media_file.filename)[1].lower()

    if file_extension in ['.mp4', '.avi', '.mov', '.mkv']:  # Video file types
        return process_video(media_path)
    elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:  # Image file types
        return process_image(media_path)
    else: 
        return jsonify({'error': 'Unsupported file type'}), 400

def process_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    deepfake_count = 0
    total_frames = 0
    confidence_scores = []
    frame_histograms = []
    deepfake_frame = None  # To store the frame where deepfake is detected

    THRESHOLD = 0.9  # Adjusted threshold
    video_frames = []  # Store frames for video playback

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        deepfake_detected, confidence_score, histograms = detect_deepfake_in_frame(frame, THRESHOLD)
        if deepfake_detected:
            deepfake_count += 1
            deepfake_frame = frame.copy()  # Save the frame where the deepfake is detected

        confidence_scores.append(confidence_score)
        frame_histograms.append(histograms)

        # Annotate the frame with the detection result
        color = (0, 255, 0) if not deepfake_detected else (0, 0, 255)
        video_frames.append(cv2.putText(frame, "Deepfake Detected" if deepfake_detected else "Real", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2))

        total_frames += 1

    cap.release()

    # Save the video with annotations
    output_video_path = os.path.join('outputs', 'output_video.mp4')
    save_annotated_video(output_video_path, video_frames)

    session['video_output_path'] = output_video_path

    return analyze_results(confidence_scores, frame_histograms, deepfake_frame)

def process_image(image_path):
    frame = cv2.imread(image_path)

    THRESHOLD = 0.7  # Adjusted threshold

    deepfake_detected, confidence_score, histograms = detect_deepfake_in_frame(frame, THRESHOLD)

    confidence_scores = [confidence_score]
    frame_histograms = [histograms]
    deepfake_frame = frame if deepfake_detected else None

    return analyze_results(confidence_scores, frame_histograms, deepfake_frame)

def detect_deepfake_in_frame(frame, threshold):
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    deepfake_detected = False
    confidence_score = 0
    histograms = None

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

            hist_b = cv2.calcHist([frame], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
            hist_r = cv2.calcHist([frame], [2], None, [256], [0, 256])
            histograms = (hist_b.flatten(), hist_g.flatten(), hist_r.flatten())

            if confidence_score > threshold:
                deepfake_detected = True
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    return deepfake_detected, confidence_score, histograms



def save_annotated_video(output_path, frames):
    height, width, layers = frames[0].shape
    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))

    for frame in frames:
        video.write(frame)

    video.release()

    # Move video to static folder for access
    static_video_path = os.path.join('static', 'outputs', os.path.basename(output_path))
    shutil.move(output_path, static_video_path)

    return static_video_path


def analyze_results(confidence_scores, frame_histograms, deepfake_frame):
    # Analyze the distribution of confidence scores
    mean_confidence = np.mean(confidence_scores)
    variance_confidence = np.var(confidence_scores)

    # Heuristic: flag as deepfake if mean confidence is high and variance is low
    deepfake_detected = mean_confidence > 0.7 and variance_confidence < 0.05

    # Save analysis data to session
    session['analysis_data'] = {
        'deepfake_detected': deepfake_detected,
        'confidence_scores': confidence_scores,
        'frame_histograms': frame_histograms,
        'deepfake_frame': deepfake_frame  # Save the frame to session
    }

    # Redirect to the analysis dashboard
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    analysis_data = session.get('analysis_data')  # Retrieve analysis data from the session
    video_output_path = session.get('video_output_path')  # Retrieve the video path

    if analysis_data is None:
        return jsonify({'error': 'Analysis data not found'}), 404

    histograms = analysis_data['frame_histograms']
    confidence_scores = analysis_data['confidence_scores']
    deepfake_detected = analysis_data['deepfake_detected']
    deepfake_frame = analysis_data['deepfake_frame']

    # Generate plots for confidence scores and histograms
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))  # Adjusted the figure size

    # Plot confidence scores
    ax[0, 0].plot(confidence_scores, color='blue')
    ax[0, 0].set_title('Confidence Scores Over Frames')
    ax[0, 0].set_xlabel('Frame')
    ax[0, 0].set_ylabel('Confidence Score')

    # Plot combined histogram of all frames for each channel
    combined_histograms = [np.mean([h[i] for h in histograms], axis=0) for i in range(3)]
    ax[0, 1].plot(combined_histograms[0], color='blue', label='Blue')
    ax[0, 1].plot(combined_histograms[1], color='green', label='Green')
    ax[0, 1].plot(combined_histograms[2], color='red', label='Red')
    ax[0, 1].set_title('Combined Histogram of All Frames')
    ax[0, 1].set_xlabel('Pixel Value')
    ax[0, 1].set_ylabel('Frequency')
    ax[0, 1].legend()

    # Show the output video path if available
    if video_output_path:
        ax[1, 0].text(0.5, 0.5, f'Video Output Path:\n{video_output_path}', 
                      horizontalalignment='center', verticalalignment='center', fontsize=10, color='white')
        ax[1, 0].axis('off')
    else:
        ax[1, 0].axis('off')  # Hide the axis if no video path is available

    plt.tight_layout()
    png_image = BytesIO()
    plt.savefig(png_image, format='png')
    png_image.seek(0)
    png_image_base64 = b64encode(png_image.getvalue()).decode('ascii')

    # Convert deepfake frame to PNG and embed in HTML
    if deepfake_frame is not None:
        _, deepfake_frame_png = cv2.imencode('.png', deepfake_frame)
        deepfake_frame_base64 = b64encode(deepfake_frame_png).decode('ascii')
    else:
        deepfake_frame_base64 = None

    # Get a random explanation from the list
    random_explanation = random.choice(fake_image_reasons)

    return render_template('dashboard.html', image_data=png_image_base64,
                           deepfake_frame_data=deepfake_frame_base64,
                           video_path=video_output_path,
                           result='Deepfake Detected' if deepfake_detected else 'Real Video',
                           confidence_scores=confidence_scores,
                           histograms=combined_histograms,
                           explanation=random_explanation)


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
