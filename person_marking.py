import os
import re
import cv2
import tempfile
import tkinter as tk
import subprocess
import zipfile
import shutil
from tkinter import filedialog
from pathlib import Path

# --- FFmpeg Setup (Reused from previous logic) ---
def _setup_ffmpeg():
    """Setup FFmpeg locally - returns ffmpeg executable path."""
    ffmpeg_folder = Path.home() / ".ffmpeg_portable"
    ffmpeg_exe = ffmpeg_folder / "bin" / "ffmpeg.exe"
    
    if ffmpeg_exe.exists():
        return str(ffmpeg_exe)
    
    print("Setting up FFmpeg (one-time only)...")
    try:
        import urllib.request
        ffmpeg_folder.mkdir(parents=True, exist_ok=True)
        zip_path = ffmpeg_folder / "ffmpeg.zip"
        url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
        
        print("Downloading FFmpeg...")
        urllib.request.urlretrieve(url, zip_path)
        
        print("Extracting FFmpeg...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(ffmpeg_folder)
        
        for root, dirs, files in os.walk(ffmpeg_folder):
            if "ffmpeg.exe" in files:
                actual_exe = Path(root) / "ffmpeg.exe"
                bin_dir = ffmpeg_folder / "bin"
                bin_dir.mkdir(exist_ok=True)
                shutil.copy(actual_exe, ffmpeg_exe)
                zip_path.unlink()
                print("✓ FFmpeg ready!")
                return str(ffmpeg_exe)
        return None
    except Exception as e:
        print(f"Error: Could not setup FFmpeg: {e}")
        return None


def _extract_audio(video_path, audio_path, ffmpeg_exe):
    """Extract audio from video using FFmpeg directly."""
    print(f"Extracting audio from video...")
    try:
        cmd = [ffmpeg_exe, "-i", video_path, "-q:a", "9", "-n", audio_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            if "moov atom not found" in result.stderr:
                raise RuntimeError("\n" + "!"*60 + "\nCORRUPTED VIDEO FILE DETECTED!\nThe video you selected is broken, incomplete, or not a valid MP4 ('moov atom not found').\nPlease try selecting a different video.\n" + "!"*60)
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
            
        if not os.path.exists(audio_path):
            raise RuntimeError(f"Audio file was not created at {audio_path}")
        print("Audio extracted successfully!")
    except Exception as e:
        raise ValueError(f"Failed to extract audio: {e}")


def find_names_in_audio(audio_path, ffmpeg_exe):
    """Transcribes audio and looks for name introductions, returning their timestamps."""
    import whisper
    
    cache_dir = Path.home() / ".cache" / "whisper"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WHISPER_CACHE"] = str(cache_dir)
    
    if ffmpeg_exe:
        ffmpeg_dir = os.path.dirname(ffmpeg_exe)
        os.environ["PATH"] = f"{ffmpeg_dir};{os.environ.get('PATH', '')}"
        
    print("Loading Whisper AI model (base)...")
    model = whisper.load_model("base", device="cpu", download_root=str(cache_dir))
    
    print("Transcribing audio to find names...")
    safe_audio_path = str(Path(audio_path).absolute()).replace("\\", "/")
    result = model.transcribe(safe_audio_path, fp16=False, language="en", verbose=False)
    
    named_segments = []
    # Common ways people introduce themselves. Captures the next word as the name.
    patterns = [
        r"(?i)my name is\s+([a-z]+)",
        r"(?i)i am\s+([a-z]+)",
        r"(?i)i\'m\s+([a-z]+)",
        r"(?i)this is\s+([a-z]+)"
    ]
    
    # Common words that might follow "I am" which are NOT names
    ignore_words = {"going", "trying", "here", "there", "not", "just", "very", "so", "a", "the", "an", "doing", "making", "looking", "in", "on", "at"}
    
    for seg in result.get('segments', []):
        text = seg['text']
        for p in patterns:
            match = re.search(p, text)
            if match:
                name = match.group(1).capitalize()
                if name.lower() not in ignore_words:
                    print(f"-> Detected name '{name}' at {seg['start']:.1f}s - {seg['end']:.1f}s")
                    named_segments.append({
                        'name': name,
                        'start': seg['start'],
                        'end': seg['end']
                    })
                    break
                    
    return named_segments


def process_video(video_path, named_segments):
    """Processes video, tracks people, and assigns names based on timestamps."""
    from ultralytics import YOLO
    
    print("Loading YOLOv8 model for tracking...")
    model_name = 'yolov8n.pt'
    if not os.path.exists(model_name):
        print(f"CRITICAL: '{model_name}' NOT FOUND. Place it in directory.")
        return
        
    model = YOLO(model_name)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_filename = "verified_people.mp4"
    
    # Use standard codec that ensures the video writes successfully
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("Error: Could not open output video writer.")
        return
        
    print(f"Processing video frames and tracking people... (Output will be saved to {output_filename})")
    
    track_names = {} # track_id -> Name
    assigned_names = set()
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        frame_count += 1
        current_time = frame_count / fps
        
        # Run YOLO Tracking
        results = model.track(frame, classes=[0], persist=True, verbose=False)[0]
        
        boxes = []
        track_ids = []
        
        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().numpy()
            
        # 1. Assign names if current time falls within a named segment
        for name_info in named_segments:
            # Adding a 2-second buffer in case of slight audio offset
            if name_info['start'] - 1.0 <= current_time <= name_info['end'] + 2.0:
                name_str = name_info['name']
                if name_str not in assigned_names and len(track_ids) > 0:
                    # Assign to the largest person on screen without a name yet
                    best_id = None
                    max_area = 0
                    for box, tid in zip(boxes, track_ids):
                        if tid not in track_names or track_names[tid] == "Unverified":
                            area = (box[2] - box[0]) * (box[3] - box[1])
                            if area > max_area:
                                max_area = area
                                best_id = tid
                                
                    if best_id is not None:
                        track_names[best_id] = name_str
                        assigned_names.add(name_str)
                        print(f"Assigned name {name_str} to person ID {best_id} at {current_time:.1f}s")

        # 2. Draw annotations
        for box, tid in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            
            # Determine name and color
            person_name = track_names.get(tid, "Unverified")
            if person_name == "Unverified":
                color = (0, 0, 255) # Red for unverified
            else:
                color = (0, 255, 0) # Green for verified
                
            # Draw Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw Label Background
            label = f"{person_name}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            
            # Draw Label Text
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        # Show progress occasionally
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({(frame_count/total_frames)*100:.1f}%)")
            
        # Display the live window
        cv2.imshow("Person Marking", frame)
        
        # WRITE the frame to the output video file
        out.write(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Important: Cleanup and save the file
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("-" * 50)
    print(f"FINISHED! Successfully saved annotated video to: {os.path.abspath(output_filename)}")


def main():
    root = tk.Tk()
    root.withdraw()
    root.lift()
    root.attributes("-topmost", True)

    print("Step 1: Please select a video file...")
    video_path = filedialog.askopenfilename(
        parent=root,
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")],
    )
    root.destroy()
    
    if not video_path:
        print("No file selected. Exiting.")
        return
        
    ffmpeg_exe = _setup_ffmpeg()
    if not ffmpeg_exe:
        print("Error: Could not setup FFmpeg. Exiting.")
        return

    temp_dir = Path(tempfile.gettempdir())
    temp_audio_path = str(temp_dir / "temp_video_audio.wav")
    
    try:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            
        _extract_audio(video_path, temp_audio_path, ffmpeg_exe)
        
        named_segments = find_names_in_audio(temp_audio_path, ffmpeg_exe)
        if not named_segments:
            print("No names were introduced in the audio (e.g. 'My name is...', 'I am...').")
        
        process_video(video_path, named_segments)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except OSError:
                pass


if __name__ == "__main__":
    main()
