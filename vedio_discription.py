import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
import os
from ultralytics import YOLO
import supervision as sv

def generate_local_scene_summary(detections_log, total_frames, fps, video_name):
    """
    Analyzes the collected detection data to provide a summary of what happened.
    """
    duration = total_frames / fps
    
    # Analyze the most common number of people seen
    counts = [d['count'] for d in detections_log]
    avg_people = sum(counts) / len(counts) if counts else 0
    max_people = max(counts) if counts else 0
    
    # Identify segments with high activity
    busy_moments = [d['time'] for d in detections_log if d['count'] > avg_people]
    
    report = [
        f"LOCAL SCENE ANALYSIS for: {video_name}",
        f"Video Duration: {duration:.2f} seconds",
        "-" * 30,
        f"Observation: On average, there were {avg_people:.1f} people present.",
        f"Peak Activity: At most, {max_people} people were seen simultaneously.",
        f"Activity Summary: The AI monitored person-based movement throughout the clip.",
        f"Scene Context: This appears to be a setting with {'high' if avg_people > 2 else 'low'} foot traffic."
    ]
    return "\n".join(report)

def analyze_video():
    """
    Prompts user for a video, analyzes every frame for people, 
    and generates a summary of the activity found.
    """
    # 1. Setup hidden Tkinter root for file selection
    root = tk.Tk()
    root.withdraw()
    root.lift()
    root.attributes("-topmost", True)
    
    print("Step 1: Please select a video file for analysis...")
    video_path = filedialog.askopenfilename(
        parent=root,
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
    )
    root.destroy()
    
    if not video_path:
        print("No file selected. Exiting.")
        return

    # 2. Load the Model
    model_name = 'yolov8n.pt'
    print(f"Step 2: Loading AI model ({model_name})...")
    
    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Open the source video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_count = 0
    detections_log = []

    print(f"Step 3: Analyzing video content...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        processed_count += 1
        current_time = processed_count / fps
        
        # Run Inference (Detecting people)
        results = model(frame, classes=[0], conf=0.35, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # Log the count for this specific moment
        detections_log.append({
            'time': round(current_time, 2),
            'count': len(detections)
        })

        # Visual feedback every 50 frames
        if processed_count % 50 == 0:
            print(f"Progress: {(processed_count/total_frames)*100:.1f}% analyzed...")

        # Display analysis preview
        box_annotator = sv.BoxAnnotator(thickness=2)
        annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
        
        cv2.putText(annotated_frame, f"Time: {current_time:.1f}s", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"People: {len(detections)}", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("AI Video Analysis", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # 4. Final Analysis Report
    if detections_log:
        report = generate_local_scene_summary(
            detections_log, 
            processed_count, 
            fps, 
            os.path.basename(video_path)
        )
        
        print("\n" + "="*50)
        print("VIDEO ANALYSIS REPORT")
        print("="*50)
        print(report)
        print("="*50)
    else:
        print("Analysis ended early or no data was collected.")

if __name__ == "__main__":
    analyze_video()