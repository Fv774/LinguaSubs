import os
import sys
from pathlib import Path
import whisper
import subprocess  # For calling ffmpeg
import torch
import threading
from tkinter import *
from tkinter import filedialog, ttk, messagebox
import queue

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

    def process_audio(self, audio_path, update_status=None):
        """Process a single audio or video file with optional status update callback"""
        if update_status:
            update_status(f"Processing: {os.path.basename(audio_path)}")

        media_suffix = Path(audio_path).suffix.lower()
        if media_suffix in ['.mp4', '.mkv', '.avi', '.mov', '.webm']:
            if update_status:
                update_status("Extracting audio from video...")
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
                if update_status:
                    update_status(f"Skipping already processed file: {os.path.basename(output_path)}")
                return True

            if update_status:
                update_status("Transcribing audio...")
            result = self.transcribe_audio(audio_path)
            if result is None:
                return False

            if update_status:
                update_status("Saving subtitles...")
            return self.save_transcript(result, output_path)
        except Exception as e:
            if update_status:
                update_status(f"Error: {str(e)}")
            print(f"Failed to process audio - {audio_path}: {e}")
            return False


class SubtitleGUI:
    """图形界面类"""

    def __init__(self, root):
        self.root = root
        self.root.title("音频字幕提取工具")
        self.root.geometry("700x500")
        self.root.resizable(True, True)

        # 设置中文字体
        self.style = ttk.Style()
        self.style.configure("TLabel", font=("SimHei", 10))
        self.style.configure("TButton", font=("SimHei", 10))
        self.style.configure("TCombobox", font=("SimHei", 10))

        # 创建主框架
        self.main_frame = ttk.Frame(root, padding="20")
        self.main_frame.pack(fill=BOTH, expand=True)

        # 文件选择部分
        self.file_frame = ttk.LabelFrame(self.main_frame, text="文件选择", padding="10")
        self.file_frame.pack(fill=X, pady=5)

        self.file_path_var = StringVar()
        self.file_path_entry = ttk.Entry(self.file_frame, textvariable=self.file_path_var, width=50)
        self.file_path_entry.pack(side=LEFT, fill=X, expand=True, padx=(0, 10))

        self.browse_file_btn = ttk.Button(self.file_frame, text="选择文件", command=self.browse_file)
        self.browse_file_btn.pack(side=LEFT, padx=(0, 5))

        self.browse_dir_btn = ttk.Button(self.file_frame, text="选择目录", command=self.browse_directory)
        self.browse_dir_btn.pack(side=LEFT)

        # 参数设置部分
        self.param_frame = ttk.LabelFrame(self.main_frame, text="参数设置", padding="10")
        self.param_frame.pack(fill=X, pady=5)

        # 模型大小
        ttk.Label(self.param_frame, text="模型大小:").grid(row=0, column=0, sticky=W, pady=5)
        self.model_size_var = StringVar(value="medium")
        self.model_size_combo = ttk.Combobox(
            self.param_frame,
            textvariable=self.model_size_var,
            values=["tiny", "base", "small", "medium", "large"],
            width=10
        )
        self.model_size_combo.grid(row=0, column=1, sticky=W, pady=5)

        # 输出格式
        ttk.Label(self.param_frame, text="输出格式:").grid(row=0, column=2, sticky=W, pady=5)
        self.output_format_var = StringVar(value="txt")
        self.output_format_combo = ttk.Combobox(
            self.param_frame,
            textvariable=self.output_format_var,
            values=["txt", "srt"],
            width=10
        )
        self.output_format_combo.grid(row=0, column=3, sticky=W, pady=5)

        # 是否包含时间戳
        self.include_timestamps_var = BooleanVar(value=True)
        ttk.Checkbutton(
            self.param_frame,
            text="包含时间戳",
            variable=self.include_timestamps_var
        ).grid(row=0, column=4, sticky=W, pady=5, padx=10)

        # 段落行数
        ttk.Label(self.param_frame, text="段落行数:").grid(row=1, column=0, sticky=W, pady=5)
        self.paragraphs_var = IntVar(value=10)
        self.paragraphs_spinbox = ttk.Spinbox(
            self.param_frame,
            from_=1, to=50,
            textvariable=self.paragraphs_var,
            width=5
        )
        self.paragraphs_spinbox.grid(row=1, column=1, sticky=W, pady=5)

        # 状态和日志部分
        self.status_frame = ttk.LabelFrame(self.main_frame, text="处理状态", padding="10")
        self.status_frame.pack(fill=BOTH, expand=True, pady=5)

        self.status_var = StringVar(value="就绪")
        ttk.Label(self.status_frame, textvariable=self.status_var).pack(anchor=W, pady=5)

        self.log_text = Text(self.status_frame, height=10, wrap=WORD)
        self.log_text.pack(fill=BOTH, expand=True, pady=5)

        self.scrollbar = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
        self.scrollbar.pack(side=RIGHT, fill=Y)
        self.log_text.config(yscrollcommand=self.scrollbar.set)

        # 底部按钮
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(fill=X, pady=10)

        self.start_btn = ttk.Button(self.button_frame, text="开始处理", command=self.start_processing)
        self.start_btn.pack(side=RIGHT, padx=(5, 0))

        self.clear_btn = ttk.Button(self.button_frame, text="清空日志", command=self.clear_log)
        self.clear_btn.pack(side=RIGHT, padx=(5, 0))

        # 线程和队列用于异步处理
        self.processing_queue = queue.Queue()
        self.root.after(100, self.check_queue)

        # 初始化日志
        self.log("欢迎使用音频字幕提取工具")
        self.log("请选择要处理的文件或目录，然后点击开始处理")

    def browse_file(self):
        """浏览并选择单个文件"""
        file_path = filedialog.askopenfilename(
            title="选择音频/视频文件",
            filetypes=[
                ("媒体文件", "*.mp3 *.wav *.ogg *.flac *.m4a *.aac *.mp4 *.mkv *.avi *.mov *.webm"),
                ("所有文件", "*.*")
            ]
        )
        if file_path:
            self.file_path_var.set(file_path)

    def browse_directory(self):
        """浏览并选择目录"""
        dir_path = filedialog.askdirectory(title="选择目录")
        if dir_path:
            self.file_path_var.set(dir_path)

    def log(self, message):
        """添加日志消息"""
        self.log_text.insert(END, message + "\n")
        self.log_text.see(END)

    def clear_log(self):
        """清空日志"""
        self.log_text.delete(1.0, END)

    def update_status(self, message):
        """更新状态消息"""
        self.status_var.set(message)
        self.log(message)

    def check_queue(self):
        """检查队列中的消息并更新UI"""
        while not self.processing_queue.empty():
            try:
                message = self.processing_queue.get(0)
                self.update_status(message)
            except queue.Empty:
                pass
        self.root.after(100, self.check_queue)

    def start_processing(self):
        """开始处理文件"""
        file_or_dir = self.file_path_var.get()

        if not file_or_dir:
            messagebox.showerror("错误", "请选择文件或目录")
            return

        if not os.path.exists(file_or_dir):
            messagebox.showerror("错误", f"路径不存在: {file_or_dir}")
            return

        # 获取参数
        model_size = self.model_size_var.get()
        output_format = self.output_format_var.get()
        include_timestamps = self.include_timestamps_var.get()
        paragraphs = self.paragraphs_var.get()

        # 禁用开始按钮
        self.start_btn.config(state=DISABLED)

        # 创建处理线程
        processing_thread = threading.Thread(
            target=self.process_files_thread,
            args=(file_or_dir, model_size, output_format, include_timestamps, paragraphs)
        )
        processing_thread.daemon = True
        processing_thread.start()

    def process_files_thread(self, file_or_dir, model_size, output_format, include_timestamps, paragraphs):
        """在单独线程中处理文件"""
        try:
            # 获取文件列表
            if os.path.isfile(file_or_dir):
                files = [file_or_dir]
            else:
                # 支持的音频和视频格式
                media_extensions = [
                    '.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac',
                    '.mp4', '.mkv', '.avi', '.mov', '.webm'
                ]
                files = [
                    os.path.join(file_or_dir, f) for f in os.listdir(file_or_dir)
                    if os.path.isfile(os.path.join(file_or_dir, f))
                       and Path(f).suffix.lower() in media_extensions
                ]

            if not files:
                self.processing_queue.put("未找到媒体文件")
                self.root.after(0, lambda: self.start_btn.config(state=NORMAL))
                return

            self.processing_queue.put(f"找到 {len(files)} 个文件")

            # 创建提取器实例
            extractor = AudioSubtitleExtractor(
                model_size=model_size,
                output_format=output_format,
                include_timestamps=include_timestamps,
                paragraphs=paragraphs
            )

            success_count = 0
            total_files = len(files)

            # 处理每个文件
            for i, audio_path in enumerate(files, 1):
                def update_progress(msg):
                    self.processing_queue.put(f"[{i}/{total_files}] {msg}")

                if extractor.process_audio(audio_path, update_progress):
                    success_count += 1

            # 处理完成
            self.processing_queue.put(
                f"处理完成: 成功={success_count}, 失败={total_files - success_count}, 总计={total_files}")

        except Exception as e:
            self.processing_queue.put(f"处理过程中发生错误: {str(e)}")
        finally:
            # 重新启用开始按钮
            self.root.after(0, lambda: self.start_btn.config(state=NORMAL))


def main():
    # 如果有命令行参数，使用命令行模式
    if len(sys.argv) > 1:
        path = sys.argv[1]
        model_size = sys.argv[2] if len(sys.argv) > 2 else "medium"
        output_format = sys.argv[3] if len(sys.argv) > 3 else "txt"
        include_timestamps = bool(int(sys.argv[4])) if len(sys.argv) > 4 else True
        paragraphs = int(sys.argv[5]) if len(sys.argv) > 5 else 10

        files = get_files_from_path(path)
        if files:
            process_files(files, model_size, output_format, include_timestamps, paragraphs)
    else:
        # 否则使用GUI模式
        root = Tk()
        app = SubtitleGUI(root)
        root.mainloop()


if __name__ == "__main__":
    main()