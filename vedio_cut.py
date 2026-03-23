import cv2
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import os
from ultralytics import YOLO
import supervision as sv

def detect_and_trim_video():
    """
    Prompts user for a video and a target count. 
    Saves a new video containing ONLY frames that match the target count.
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

    # Ask the user how many people they want to keep
    target_count = simpledialog.askinteger(
        "Target Person Count", 
        "Enter the exact number of people you want to see per frame:",
        parent=root,
        minvalue=0,
        maxvalue=100
    )
    
    root.destroy()
    
    if target_count is None:
        print("No target count provided. Exiting.")
        return

    # 2. Load the Model
    model_name = 'yolov8n.pt'
    print(f"Step 2: Loading AI model ({model_name})...")
    
    if not os.path.exists(model_name):
        print(f"ERROR: '{model_name}' not found in current directory.")
        return

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

    # Get video properties for the Output Writer
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup Output Video Writer
    output_filename = "filtered_moments.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    print(f"Step 3: Processing... Creating a new video with frames containing EXACTLY {target_count} people.")
    print("Press 'q' to stop early and save what has been processed.")

    saved_frames_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        processed_count += 1
        
        # Run Inference (People only = class 0)
        results = model(frame, classes=[0], conf=0.35, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        current_person_count = len(detections)

        # 4. Filter Logic: Only write to output if count matches target
        if current_person_count == target_count:
            # Annotate the frame so the user knows why it was kept
            box_annotator = sv.BoxAnnotator(thickness=2)
            annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
            
            cv2.putText(annotated_frame, f"Match Found: {target_count} People", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            out.write(annotated_frame)
            saved_frames_count += 1

        # Visual feedback for progress
        if processed_count % 30 == 0:
            progress = (processed_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% | Saved {saved_frames_count} matching frames...")

        # Display preview
        cv2.imshow("Filtering Video... (Preview)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("-" * 50)
    print("FINISHED!")
    if saved_frames_count > 0:
        print(f"Success: {saved_frames_count} frames matched your criteria.")
        print(f"The new video has been saved as: {os.path.abspath(output_filename)}")
    else:
        print("No frames were found that matched your exact person count.")
        # Delete empty file if nothing was found
        if os.path.exists(output_filename):
            os.remove(output_filename)
    print("-" * 50)

if __name__ == "__main__":
    detect_and_trim_video()