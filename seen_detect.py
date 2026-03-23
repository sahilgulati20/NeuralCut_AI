import cv2
import tkinter as tk
from tkinter import filedialog
import os
from ultralytics import YOLO
import supervision as sv
import sys

def detect_people_in_video():
    """
    Prompts user to upload a video, then uses YOLOv8 and 
    Supervision to detect and count people. Saves a second-by-second report.
    """
    # 1. Setup hidden Tkinter root for file dialog
    root = tk.Tk()
    root.withdraw()
    root.lift()
    root.attributes("-topmost", True)
    
    print("Step 1: Please select a video file...")
    video_path = filedialog.askopenfilename(
        parent=root,
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
    )
    root.destroy()
    
    if not video_path:
        print("No file selected. Exiting.")
        return

    # 2. Load the Model using Ultralytics
    model_name = 'yolov8n.pt'
    print(f"Step 2: Loading AI model ({model_name})...")
    
    if not os.path.exists(model_name):
        print("\n" + "!"*50)
        print(f"CRITICAL ERROR: '{model_name}' NOT FOUND.")
        print(f"Please place the model file in: {os.getcwd()}")
        print("!"*50 + "\n")
        return

    try:
        model = YOLO(model_name)
        print("Success: Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Setup Supervision Annotators
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    # 4. Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties for time calculation
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30 # Fallback if FPS detection fails

    print("Step 3: Processing video... Press 'q' to stop.")
    print("-" * 40)
    print(f"{'Time (s)':<10} | {'People Detected':<15}")
    print("-" * 40)

    total_max_people = 0
    time_log = [] # To store (timestamp, count) for the final report

    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Calculate current timestamp in seconds
        current_time_sec = frame_count / fps
        frame_count += 1

        # 5. Run Inference
        results = model(frame, classes=[0], conf=0.35, verbose=False)[0]

        # 6. Convert to Supervision Detections
        detections = sv.Detections.from_ultralytics(results)
        current_count = len(detections)

        # 7. Update person counting logic
        if current_count > total_max_people:
            total_max_people = current_count

        # 8. Log every 1 second (to avoid terminal spamming)
        if frame_count % int(fps) == 0:
            timestamp_str = f"{current_time_sec:.1f}s"
            print(f"{timestamp_str:<10} | {current_count:<15}")
            time_log.append((timestamp_str, current_count))

        # 9. Annotate the frame
        annotated_frame = box_annotator.annotate(
            scene=frame.copy(), 
            detections=detections
        )
        
        labels = [f"Person {conf:.2f}" for conf in detections.confidence]
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, 
            detections=detections,
            labels=labels
        )
        
        # Add visual overlay for time and count
        cv2.putText(annotated_frame, f"Time: {current_time_sec:.1f}s", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Count: {current_count}", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 10. Display
        cv2.imshow("Person Detection Log", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 11. Final Cleanup and Report Saving
    cap.release()
    cv2.destroyAllWindows()

    # Save to file
    report_file = "detection_report.txt"
    with open(report_file, "w") as f:
        f.write("PEOPLE DETECTION REPORT\n")
        f.write("="*30 + "\n")
        f.write(f"Video File: {os.path.basename(video_path)}\n")
        f.write(f"Peak Count: {total_max_people}\n")
        f.write("-" * 30 + "\n")
        f.write(f"{'Timestamp':<15} | {'Count':<10}\n")
        for ts, count in time_log:
            f.write(f"{ts:<15} | {count:<10}\n")

    print("-" * 40)
    print(f"Analysis Complete!")
    print(f"Max people at once: {total_max_people}")
    print(f"Full report saved to: {os.path.abspath(report_file)}")
    print("-" * 40)

if __name__ == "__main__":
    detect_people_in_video()