import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response, send_file
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import base64
from io import BytesIO
from PIL import Image
import tempfile
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load YOLO model
try:
    model = YOLO('best.pt')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Global variable for webcam
camera = None

def init_camera():
    """Initialize camera"""
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        # Allow camera to warm up
        time.sleep(2)
    return camera

def release_camera():
    """Release camera resources"""
    global camera
    if camera is not None:
        camera.release()
        camera = None

def process_image(image, output_path=None):
    """Process image and return detection results"""
    try:
        if model is None:
            return False, [], 0, image
            
        # Run YOLO inference
        results = model(image)
        
        # Plot results on image
        plotted_image = results[0].plot()
        
        # Convert BGR to RGB
        plotted_image_rgb = cv2.cvtColor(plotted_image, cv2.COLOR_BGR2RGB)
        
        # Save result if output path provided
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(plotted_image_rgb, cv2.COLOR_RGB2BGR))
        
        # Get detection information
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = result.names[cls]
                    detections.append({
                        'label': label,
                        'confidence': round(conf, 2),
                        'class': cls
                    })
        
        return True, detections, len(detections), plotted_image_rgb
    
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return False, [], 0, image

def process_image_file(image_path, output_path):
    """Process image file and return detection results"""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return False, [], 0
            
        success, detections, total_detections, processed_image = process_image(image, output_path)
        return success, detections, total_detections
    
    except Exception as e:
        print(f"❌ Error processing image file: {e}")
        return False, [], 0

def process_video(video_path, output_path):
    """Process video and return detection results"""
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return False, 0
            
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        total_monkeys = 0
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run YOLO inference
            success, detections, frame_monkeys, processed_frame = process_image(frame)
            
            # Write frame to output video
            out.write(cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
            
            # Count monkeys in current frame
            total_monkeys = max(total_monkeys, frame_monkeys)
            
            frame_count += 1
            
            # Progress update every 50 frames
            if frame_count % 50 == 0:
                print(f"📊 Processed {frame_count} frames...")
        
        cap.release()
        out.release()
        
        return True, total_monkeys
    
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        return False, 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image-detection')
def image_detection():
    return render_template('image.html')

@app.route('/video-detection')
def video_detection():
    return render_template('video.html')

@app.route('/webcam-detection')
def webcam_detection():
    return render_template('webcam.html')

@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            
            # Generate output filename
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_result{ext}"
            output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
            
            # Process image
            success, detections, total_detections = process_image_file(upload_path, output_path)
            
            if success:
                return jsonify({
                    'success': True,
                    'result_image': f'/static/results/{output_filename}',
                    'detections': detections,
                    'total_detections': total_detections,
                    'message': f'🎉 Successfully detected {total_detections} monkey(s)!'
                })
            else:
                return jsonify({'error': 'Error processing image'}), 500
                
        except Exception as e:
            return jsonify({'error': f'Error: {str(e)}'}), 500

@app.route('/upload-video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            
            # Generate output filename
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_result{ext}"
            output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
            
            # Process video
            success, total_monkeys = process_video(upload_path, output_path)
            
            if success:
                message = f'🎉 Video processing complete! Maximum monkeys detected in any frame: {total_monkeys}'
                return jsonify({
                    'success': True,
                    'result_video': f'/static/results/{output_filename}',
                    'total_monkeys': total_monkeys,
                    'message': message
                })
            else:
                return jsonify({'error': 'Error processing video'}), 500
                
        except Exception as e:
            return jsonify({'error': f'Error: {str(e)}'}), 500

def generate_frames():
    """Generate frames from webcam with YOLO detection"""
    camera = init_camera()
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Run YOLO inference
            if model:
                success, detections, total_detections, processed_frame = process_image(frame)
                frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
            else:
                # If model not available, just use original frame
                frame = frame
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect-webcam', methods=['POST'])
def detect_webcam():
    """Take a snapshot from webcam and process it"""
    try:
        camera = init_camera()
        
        # Capture frame
        success, frame = camera.read()
        
        if not success:
            return jsonify({'error': '❌ Failed to capture image from webcam. Please check your camera connection.'}), 500
        
        # Process the frame
        success, detections, total_detections, processed_frame = process_image(frame)
        
        if not success:
            return jsonify({'error': '❌ Error processing webcam image'}), 500
        
        # Convert to PIL Image
        pil_image = Image.fromarray(processed_frame)
        
        # Save to bytes
        img_io = BytesIO()
        pil_image.save(img_io, 'JPEG', quality=85)
        img_io.seek(0)
        
        # Convert to base64 for response
        img_base64 = base64.b64encode(img_io.getvalue()).decode('ascii')
        
        # Prepare response message
        if total_detections > 0:
            message = f'🎉 Monkey Detected! Found {total_detections} monkey(s) in the image.'
        else:
            message = '❌ No monkeys detected in the image.'
        
        return jsonify({
            'success': True,
            'image': f'data:image/jpeg;base64,{img_base64}',
            'detections': detections,
            'total_detections': total_detections,
            'message': message
        })
        
    except Exception as e:
        print(f"❌ Error in webcam detection: {e}")
        return jsonify({'error': f'❌ Camera error: {str(e)}'}), 500

@app.route('/start-webcam', methods=['POST'])
def start_webcam():
    """Initialize webcam"""
    try:
        camera = init_camera()
        success, frame = camera.read()
        if success:
            return jsonify({'success': True, 'message': '✅ Webcam started successfully'})
        else:
            return jsonify({'success': False, 'error': '❌ Failed to start webcam'})
    except Exception as e:
        return jsonify({'success': False, 'error': f'❌ Webcam error: {str(e)}'})

@app.route('/stop-webcam', methods=['POST'])
def stop_webcam():
    """Release webcam resources"""
    try:
        release_camera()
        return jsonify({'success': True, 'message': '✅ Webcam stopped successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': f'❌ Error stopping webcam: {str(e)}'})

@app.teardown_appcontext
def close_camera(error):
    """Ensure camera is released when app closes"""
    release_camera()

if __name__ == '__main__':
    print("🚀 Starting Monkey Detection System...")
    print("📷 Initializing camera...")
    
    # Test camera on startup
    try:
        test_cam = cv2.VideoCapture(0)
        if test_cam.isOpened():
            print("✅ Camera initialized successfully")
            test_cam.release()
        else:
            print("❌ Warning: Could not initialize camera")
    except:
        print("❌ Warning: Camera test failed")
    
    app.run(debug=True, host='0.0.0.0', port=5000)