import cv2
import tkinter as tk
from tkinter import filedialog, simpledialog
import os
from ultralytics import YOLO
import supervision as sv
import sys

def detect_people_in_video():
    """
    Prompts user to upload a video and specify a target number of people.
    Detects people and alerts the user when the target count is met.
    """
    # 1. Setup hidden Tkinter root for dialogs
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
    
    if not video_path:
        print("No file selected. Exiting.")
        root.destroy()
        return

    # Ask the user how many people they are looking for
    target_count = simpledialog.askinteger(
        "Target Count", 
        "How many people are you looking for in a single frame?",
        parent=root,
        minvalue=1,
        maxvalue=100
    )
    
    root.destroy()
    
    if target_count is None:
        print("No target count provided. Exiting.")
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
    if fps == 0: fps = 30 

    print(f"Step 3: Searching for moments with EXACTLY {target_count} people...")
    print("-" * 50)
    print(f"{'Time (s)':<10} | {'Status':<25}")
    print("-" * 50)

    total_max_people = 0
    target_moments = [] # To store timestamps where target was met
    time_log = [] 

    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        current_time_sec = frame_count / fps
        frame_count += 1

        # 5. Run Inference
        results = model(frame, classes=[0], conf=0.35, verbose=False)[0]

        # 6. Convert to Supervision Detections
        detections = sv.Detections.from_ultralytics(results)
        current_count = len(detections)

        if current_count > total_max_people:
            total_max_people = current_count

        # 7. Check if current count matches target count
        is_target_met = (current_count == target_count)
        
        # Log to terminal every 1 second or when target is met
        if is_target_met or (frame_count % int(fps) == 0):
            timestamp_str = f"{current_time_sec:.1f}s"
            status = f"MATCH FOUND ({current_count})" if is_target_met else f"Count: {current_count}"
            print(f"{timestamp_str:<10} | {status:<25}")
            
            if is_target_met:
                target_moments.append(timestamp_str)
            
            if frame_count % int(fps) == 0:
                time_log.append((timestamp_str, current_count))

        # 8. Annotate the frame
        # Change box color to Green if target is met, otherwise standard
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
        
        # Add visual overlay
        color = (0, 255, 0) if is_target_met else (255, 255, 255)
        alert_text = "TARGET MET!" if is_target_met else ""
        
        cv2.putText(annotated_frame, f"Time: {current_time_sec:.1f}s", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Count: {current_count}", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(annotated_frame, alert_text, (20, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        # 9. Display
        cv2.imshow("Person Search Mode", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # 10. Save Detailed Report
    report_file = "search_report.txt"
    with open(report_file, "w") as f:
        f.write("SEARCH REPORT\n")
        f.write("="*30 + "\n")
        f.write(f"Target Count: {target_count} people\n")
        f.write(f"Max detected: {total_max_people}\n")
        f.write("-" * 30 + "\n")
        if target_moments:
            f.write(f"MOMENTS WHERE {target_count} PEOPLE WERE PRESENT:\n")
            # Remove duplicates for the report if multiple frames per second match
            for m in sorted(list(set(target_moments))):
                f.write(f"- {m}\n")
        else:
            f.write(f"No moments found with exactly {target_count} people.\n")

    print("-" * 50)
    print(f"Search Complete!")
    if target_moments:
        print(f"Found {target_count} people at these times: {', '.join(list(set(target_moments))[:5])}...")
    else:
        print(f"Did not find any moments with exactly {target_count} people.")
    print(f"Detailed report saved to: {report_file}")
    print("-" * 50)

if __name__ == "__main__":
    detect_people_in_video()