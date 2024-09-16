import cv2
import pyaudio
import wave
import boto3
import time
import os
import threading
from pydub import AudioSegment

# Initialize S3 client with credentials
s3 = boto3.client(
    's3',
    aws_access_key_id='your_Access-key-id',
    aws_secret_access_key='your_secret-access-key',
    region_name='your-region'  # Optional: specify the region
)

# S3 bucket name
BUCKET_NAME = 'your-bucket-name'

# Load the pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize video capture
cap = cv2.VideoCapture(0)

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
audio = pyaudio.PyAudio()

# Recording flag and management variables
recording = False
video_thread = None
audio_thread = None
stop_event = threading.Event()

# Function to record audio
def record_audio_to_file(audio_filename, stop_event):
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    while not stop_event.is_set():
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
    wf = wave.open(audio_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    stream.stop_stream()
    stream.close()
    print(f"Audio saved as {audio_filename}")

# Function to upload file to S3
def upload_to_s3(file_name, bucket):
    try:
        s3.upload_file(file_name, bucket, os.path.basename(file_name))
        print(f"File uploaded successfully to {bucket}/{os.path.basename(file_name)}")
    except Exception as e:
        print(f"Error uploading file: {e}")

# Function to combine audio and video
def combine_audio_video(video_filename, audio_filename, output_filename):
    video_clip = cv2.VideoCapture(video_filename)
    audio_clip = AudioSegment.from_wav(audio_filename)

    # Load the video and audio streams separately
    frame_width = int(video_clip.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_clip.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_clip.get(cv2.CAP_PROP_FPS)

    frame_count = int(video_clip.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = len(audio_clip)

    if duration == 0:
        print(f"Error: Audio duration is zero for {audio_filename}")
        return

    frames_per_ms = frame_count / duration  # Frame count per millisecond of audio

    # Set up video writer to write output file
    out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Write frames with the correct audio timings
    for i in range(frame_count):
        ret, frame = video_clip.read()
        if ret:
            out.write(frame)

    out.release()
    video_clip.release()
    print(f"Video and audio combined into {output_filename}")

# Function to start recording video
def record_video(video_filename, stop_event):
    out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (640, 480))
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            out.write(frame)
    out.release()
    print(f"Video saved as {video_filename}")

# Start the video capture loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Start recording when a face is detected
    if len(faces) > 0:
        if not recording:
            print("Face detected. Starting recording...")

            # Filenames for video, audio, and final output
            video_filename = f"recording_{int(time.time())}.mp4"
            audio_filename = f"audio_{int(time.time())}.wav"
            output_filename = f"final_{int(time.time())}.mp4"

            # Reset the stop event for new threads
            stop_event.clear()

            # Start video recording in a background thread
            video_thread = threading.Thread(target=record_video, args=(video_filename, stop_event))
            video_thread.start()

            # Start audio recording in a background thread
            audio_thread = threading.Thread(target=record_audio_to_file, args=(audio_filename, stop_event))
            audio_thread.start()

            recording = True

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        # Stop recording when no face is detected
        if recording:
            print("No face detected. Stopping recording...")

            # Set stop event to terminate threads
            stop_event.set()

            # Wait for threads to finish
            video_thread.join()
            audio_thread.join()

            # Combine video and audio into a single file
            combine_audio_video(video_filename, audio_filename, output_filename)

            # Upload the final output to S3
            upload_to_s3(output_filename, BUCKET_NAME)

            # Clean up local files
            os.remove(video_filename)
            os.remove(audio_filename)
            os.remove(output_filename)

            recording = False

    # Display the video frame with face detection
    cv2.imshow('Face Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources when exiting
cap.release()
cv2.destroyAllWindows()
audio.terminate()