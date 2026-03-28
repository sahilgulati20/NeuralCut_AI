import os
import re
import tempfile
import tkinter as tk
import subprocess
import zipfile
import shutil
from collections import Counter
from tkinter import filedialog
from pathlib import Path


def _setup_ffmpeg():
    """Setup FFmpeg locally if not found in PATH - returns ffmpeg executable path."""
    ffmpeg_folder = Path.home() / ".ffmpeg_portable"
    ffmpeg_exe = ffmpeg_folder / "bin" / "ffmpeg.exe"
    
    if ffmpeg_exe.exists():
        return str(ffmpeg_exe)
    
    print("Downloading FFmpeg (one-time setup)...")
    try:
        import urllib.request
        
        ffmpeg_folder.mkdir(parents=True, exist_ok=True)
        zip_path = ffmpeg_folder / "ffmpeg.zip"
        
        # Download FFmpeg portable build for Windows
        url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
        
        print("Downloading FFmpeg... (this may take a minute)")
        urllib.request.urlretrieve(url, zip_path)
        
        print("Extracting FFmpeg...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(ffmpeg_folder)
        
        # Find the extracted ffmpeg.exe
        for root, dirs, files in os.walk(ffmpeg_folder):
            if "ffmpeg.exe" in files:
                actual_exe = Path(root) / "ffmpeg.exe"
                # Copy to bin directory for consistency
                bin_dir = ffmpeg_folder / "bin"
                bin_dir.mkdir(exist_ok=True)
                shutil.copy(actual_exe, ffmpeg_exe)
                
                # Clean up zip
                zip_path.unlink()
                
                print("FFmpeg setup complete!")
                return str(ffmpeg_exe)
        
        print("Warning: Could not find ffmpeg.exe in downloaded archive")
        return None
        
    except Exception as e:
        print(f"Warning: Could not auto-download FFmpeg: {e}")
        return None


def _check_dependencies():
    """Check required Python packages."""
    try:
        import whisper
    except ImportError:
        print("Missing: pip install openai-whisper")
        return None

    return whisper


def _select_video_file():
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
    return video_path


def _extract_audio(video_path, audio_path, ffmpeg_exe):
    """Extract audio from video using FFmpeg directly."""
    print(f"Extracting audio from video...")
    
    try:
        cmd = [
            ffmpeg_exe,
            "-i", video_path,
            "-q:a", "9",
            "-n",
            audio_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            if "moov atom not found" in result.stderr:
                raise RuntimeError("\n" + "!"*60 + "\nCORRUPTED VIDEO FILE DETECTED!\nThe video you selected is broken, incomplete, or not a valid MP4 ('moov atom not found').\nPlease try selecting a different video.\n" + "!"*60)
            raise RuntimeError(f"FFmpeg error: {result.stderr}")
        
        if not os.path.exists(audio_path):
            raise RuntimeError("Audio file was not created")
        
        # Get duration using ffmpeg
        probe_cmd = [
            ffmpeg_exe,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1:nokey=1",
            video_path
        ]
        
        try:
            duration_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
            duration_seconds = float(duration_result.stdout.strip())
        except:
            duration_seconds = 0
        
        print(f"Audio extracted successfully!")
        return duration_seconds
        
    except Exception as e:
        raise ValueError(f"Failed to extract audio: {e}")


def _transcribe_audio(audio_path, whisper_module, model_size="base", ffmpeg_exe=None):
    print(f"Step 3: Loading Whisper model ({model_size})...")
    
    # Set model cache directory
    cache_dir = Path.home() / ".cache" / "whisper"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WHISPER_CACHE"] = str(cache_dir)
    
    # Crucial: whisper hardcodes calling 'ffmpeg' inside its internal methods
    # Add our local ffmpeg directory to PATH so whisper can find it
    if ffmpeg_exe:
        ffmpeg_dir = os.path.dirname(ffmpeg_exe)
        os.environ["PATH"] = f"{ffmpeg_dir};{os.environ.get('PATH', '')}"
    
    # Convert audio path to absolute Windows path
    audio_path = str(Path(audio_path).absolute())
    print(f"Audio file: {audio_path}")
    print(f"Audio exists: {os.path.exists(audio_path)}")
    print(f"Audio size: {os.path.getsize(audio_path)} bytes")
    
    try:
        # Force CPU to avoid GPU issues
        model = whisper_module.load_model(model_size, device="cpu", download_root=str(cache_dir))
    except Exception as e:
        print(f"Error loading model: {e}")
        raise RuntimeError(f"Failed to load Whisper model: {e}")

    print("Step 4: Transcribing audio (this may take a minute)...")
    try:
        # Use absolute path with forward slashes for cross-platform compatibility
        safe_audio_path = audio_path.replace("\\", "/")
        result = model.transcribe(safe_audio_path, fp16=False, language="en", verbose=False)
        return result.get("text", "").strip()
    except Exception as e:
        print(f"Error transcribing: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Failed to transcribe audio: {e}")


def _split_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


def _summarize_text(text, max_sentences=4):
    """Simple extractive summary based on word-frequency sentence scoring."""
    sentences = _split_sentences(text)
    if not sentences:
        return "No spoken content detected."

    stop_words = {
        "the", "is", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "that", "this",
        "it", "as", "at", "be", "are", "was", "were", "from", "by", "we", "you", "i", "they", "he",
        "she", "them", "his", "her", "our", "your", "their", "but", "if", "then", "so", "because", "about",
    }

    words = re.findall(r"[a-zA-Z']+", text.lower())
    filtered_words = [w for w in words if w not in stop_words and len(w) > 2]
    freq = Counter(filtered_words)

    if not freq:
        return " ".join(sentences[:max_sentences])

    sentence_scores = []
    for sentence in sentences:
        sentence_words = re.findall(r"[a-zA-Z']+", sentence.lower())
        score = sum(freq.get(w, 0) for w in sentence_words)
        sentence_scores.append((sentence, score))

    top = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:max_sentences]
    top_sentences_set = {s for s, _ in top}

    ordered_summary = [s for s in sentences if s in top_sentences_set][:max_sentences]
    return " ".join(ordered_summary)


def _analyze_text(transcript, duration_seconds):
    words = re.findall(r"\b\w+\b", transcript)
    word_count = len(words)

    minutes = duration_seconds / 60 if duration_seconds else 0
    speaking_rate = (word_count / minutes) if minutes > 0 else 0

    keywords = [
        w for w in re.findall(r"[a-zA-Z']+", transcript.lower())
        if len(w) > 3 and w not in {"this", "that", "with", "from", "have", "were", "your", "about"}
    ]
    top_keywords = Counter(keywords).most_common(8)

    return {
        "duration_seconds": duration_seconds,
        "word_count": word_count,
        "speaking_rate_wpm": speaking_rate,
        "top_keywords": top_keywords,
        "summary": _summarize_text(transcript),
    }


def _save_outputs(video_path, transcript, analysis):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    transcript_file = f"{base_name}_transcript.txt"
    analysis_file = f"{base_name}_audio_analysis.txt"

    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write("AUDIO TRANSCRIPT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Video: {os.path.basename(video_path)}\n\n")
        f.write(transcript if transcript else "No speech detected.")

    with open(analysis_file, "w", encoding="utf-8") as f:
        f.write("AUDIO ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Video: {os.path.basename(video_path)}\n")
        f.write(f"Duration (seconds): {analysis['duration_seconds']:.2f}\n")
        f.write(f"Total Words: {analysis['word_count']}\n")
        f.write(f"Estimated Speaking Rate (WPM): {analysis['speaking_rate_wpm']:.1f}\n")
        f.write("\nTop Keywords:\n")

        if analysis["top_keywords"]:
            for word, count in analysis["top_keywords"]:
                f.write(f"- {word}: {count}\n")
        else:
            f.write("- No keywords extracted\n")

        f.write("\nSummary:\n")
        f.write(analysis["summary"] + "\n")

    return transcript_file, analysis_file


def transcribe_and_analyze_video_audio(model_size="base"):
    """Extract audio, transcribe, and generate summary."""
    whisper_module = _check_dependencies()
    if not whisper_module:
        return

    ffmpeg_exe = _setup_ffmpeg()
    if not ffmpeg_exe:
        print("Error: Could not setup FFmpeg")
        return

    video_path = _select_video_file()
    if not video_path:
        print("No file selected. Exiting.")
        return

    print("Step 2: Extracting audio from video...")
    temp_audio_path = None

    try:
        # Create temp file in system temp directory (more reliable)
        temp_dir = Path(tempfile.gettempdir())
        temp_audio_path = str(temp_dir / "whisper_audio.wav")
        
        print(f"Using temp directory: {temp_dir}")
        
        # Remove old temp file if exists
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

        duration_seconds = _extract_audio(video_path, temp_audio_path, ffmpeg_exe)
        
        # Verify audio file was created
        if not os.path.exists(temp_audio_path):
            raise RuntimeError(f"Audio file was not created at {temp_audio_path}")
        
        file_size = os.path.getsize(temp_audio_path)
        print(f"Audio file created: {file_size} bytes")
        
        transcript = _transcribe_audio(temp_audio_path, whisper_module, model_size=model_size, ffmpeg_exe=ffmpeg_exe)

        print("Step 5: Analyzing transcript and generating summary...")
        analysis = _analyze_text(transcript, duration_seconds)

        transcript_file, analysis_file = _save_outputs(video_path, transcript, analysis)

        print("\n" + "=" * 60)
        print("AUDIO ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Transcript file: {os.path.abspath(transcript_file)}")
        print(f"Analysis file:   {os.path.abspath(analysis_file)}")
        print("\nQuick Summary:")
        print(analysis["summary"])
        print("=" * 60)

    except Exception as exc:
        print(f"Error: {exc}")
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
                print("Cleaned up temporary files.")
            except OSError as e:
                print(f"Warning: Could not remove temp file: {e}")


if __name__ == "__main__":
    transcribe_and_analyze_video_audio(model_size="base")
