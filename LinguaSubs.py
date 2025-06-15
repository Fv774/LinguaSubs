import os
import sys
from pathlib import Path
import whisper
import subprocess  # For calling ffmpeg
import torch

# Try to import OpenCC for Traditional-Simplified Chinese conversion
try:
    from opencc import OpenCC
except ImportError:
    print("Warning: opencc-python-reimplemented not installed. "
          "Traditional-Simplified Chinese conversion will not be available. "
          "Please run 'pip install opencc-python-reimplemented' to enable this feature.")
    OpenCC = None


class AudioSubtitleExtractor:
    """Main class for extracting subtitles from audio files"""

    def __init__(self, model_size="medium", output_format="txt",
                 include_timestamps=True, paragraphs=10):
        self.model_size = model_size
        self.output_format = output_format
        self.include_timestamps = include_timestamps
        self.paragraphs = paragraphs
        self.whisper_model = None
        # Initialize Traditional-Simplified Chinese converter
        self.cc = OpenCC('t2s') if OpenCC else None

    def load_model(self):
        """Load the specified Whisper model"""
        try:
            print(f"Loading model: {self.model_size}")
            # Force loading model configuration prioritizing Simplified Chinese (if available)
            self.whisper_model = whisper.load_model(self.model_size)
            print(f"Model loaded: {self.model_size}")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def transcribe_audio(self, audio_path):
        """Transcribe the given audio file"""
        if self.whisper_model is None:
            if not self.load_model():
                return None

        try:
            print(f"Transcribing audio...")
            # Explicitly specify Simplified Chinese to avoid misclassification of Traditional Chinese
            result = self.whisper_model.transcribe(
                audio_path,
                task="transcribe",
                language="chinese",  # Specifically for Simplified Chinese
                fp16=torch.cuda.is_available()
            )
            # Verify detected language
            detected_lang = result.get('language', 'unknown')
            print(f"Detected language: {detected_lang}")

            # If Traditional Chinese is detected, force conversion (fallback)
            if detected_lang.lower().startswith('chinese (traditional)') and self.cc:
                for seg in result['segments']:
                    seg['text'] = self.cc.convert(seg['text'])
            return result
        except Exception as e:
            print(f"Audio transcription failed - {audio_path}: {e}")
            return None

    def format_timestamp(self, seconds: float, always_include_hours: bool = False):
        """Format timestamp in SRT format"""
        assert seconds >= 0, f"Non-negative timestamp required: {seconds}"
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        hours_marker = f"{int(hours):02d}:" if always_include_hours or hours > 0 else ""
        return f"{hours_marker}{int(minutes):02d}:{int(seconds):02d},{int((seconds % 1) * 1000):03d}"

    def extract_audio_from_video(self, video_path):
        """Extract audio from video file using ffmpeg"""
        audio_path = str(Path(video_path).with_suffix(".wav"))
        if not os.path.exists(audio_path):
            print(f"Extracting audio from video: {video_path}")
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-i", video_path, "-vn",
                        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except subprocess.CalledProcessError as e:
                print(f"Failed to extract audio: {e}")
                return None
        return audio_path

    def save_transcript(self, result, output_path):
        """Save transcription result to file in specified format"""
        try:
            if self.output_format.lower() == "txt":
                with open(output_path, "w", encoding="utf-8") as f:
                    segments = result["segments"]
                    for i, segment in enumerate(segments):
                        if self.include_timestamps:
                            start_time = self.format_timestamp(segment["start"])
                            f.write(f"[{start_time}] ")
                        # Convert again to ensure Simplified Chinese (prevent model misclassification)
                        text = segment['text'].strip()
                        if self.cc:
                            text = self.cc.convert(text)
                        f.write(f"{text}\n")
                        if (i + 1) % self.paragraphs == 0:
                            f.write("\n")
                    f.write("\n")
            elif self.output_format.lower() == "srt":
                with open(output_path, "w", encoding="utf-8") as f:
                    segments = result["segments"]
                    for i, segment in enumerate(segments, start=1):
                        f.write(f"{i}\n")
                        start_time = self.format_timestamp(segment["start"])
                        end_time = self.format_timestamp(segment["end"])
                        f.write(f"{start_time} --> {end_time}\n")
                        text = segment['text'].strip()
                        if self.cc:
                            text = self.cc.convert(text)
                        f.write(f"{text}\n\n")
            else:
                print(f"Unsupported output format: {self.output_format}")
                return False
            print(f"Subtitles saved: {output_path}")
            return True
        except Exception as e:
            print(f"Failed to save subtitles - {output_path}: {e}")
            return False

    def process_audio(self, audio_path):
        """Process a single audio or video file"""
        media_suffix = Path(audio_path).suffix.lower()
        if media_suffix in ['.mp4', '.mkv', '.avi', '.mov', '.webm']:
            extracted = self.extract_audio_from_video(audio_path)
            if not extracted:
                return False
            audio_path = extracted

        try:
            audio_dir = os.path.dirname(audio_path)
            base_name = Path(audio_path).stem
            output_ext = "txt" if self.output_format.lower() == "txt" else "srt"
            output_path = os.path.join(audio_dir, f"{base_name}.{output_ext}")

            if os.path.exists(output_path):
                print(f"Skipping already processed file: {output_path}")
                return True

            print(f"\nProcessing audio: {audio_path}")
            result = self.transcribe_audio(audio_path)
            if result is None:
                return False
            return self.save_transcript(result, output_path)
        except Exception as e:
            print(f"Failed to process audio - {audio_path}: {e}")
            return False


def process_files(files, model_size="medium", output_format="txt", include_timestamps=True, paragraphs=10):
    extractor = AudioSubtitleExtractor(
        model_size=model_size,
        output_format=output_format,
        include_timestamps=include_timestamps,
        paragraphs=paragraphs
    )
    success_count = 0
    total_files = len(files)

    for i, audio_path in enumerate(files, 1):
        print(f"Processing file {i}/{total_files}: {os.path.basename(audio_path)}")
        if extractor.process_audio(audio_path):
            success_count += 1

    print(f"Processing completed: Success={success_count}, Failed={total_files - success_count}, Total={total_files}")


def get_files_from_path(path):
    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        # Support both audio and video formats
        media_extensions = [
            '.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac',
            '.mp4', '.mkv', '.avi', '.mov', '.webm'
        ]
        return [
            os.path.join(path, f) for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
               and Path(f).suffix.lower() in media_extensions
        ]
    else:
        print(f"Invalid path: {path}")
        return []


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <file_or_directory> [model_size] [output_format] [include_timestamps] [paragraphs]")
        sys.exit(1)

    path = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else "medium"
    output_format = sys.argv[3] if len(sys.argv) > 3 else "txt"
    include_timestamps = bool(int(sys.argv[4])) if len(sys.argv) > 4 else True
    paragraphs = int(sys.argv[5]) if len(sys.argv) > 5 else 10

    files = get_files_from_path(path)
    if files:
        process_files(files, model_size, output_format, include_timestamps, paragraphs)
