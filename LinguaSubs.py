import os
import sys
from pathlib import Path
import whisper
import subprocess  # 用于调用ffmpeg处理音视频
import torch

# 尝试导入OpenCC用于繁简中文转换
try:
    from opencc import OpenCC
except ImportError:
    print("警告: 未安装opencc-python-reimplemented。"
          "繁体中文到简体中文的转换功能将不可用。"
          "请运行 'pip install opencc-python-reimplemented' 启用此功能。")
    OpenCC = None


class AudioSubtitleExtractor:
    """从音频文件中提取字幕的主类
    
    该类处理使用Whisper ASR模型进行音频处理、转录和字幕生成的所有方面
    """

    def __init__(self, model_size="medium", output_format="txt",
                 include_timestamps=True, paragraphs=10):
        """使用用户指定的参数初始化字幕提取器
        
        参数:
            model_size (str): 要使用的Whisper模型大小 (tiny/base/small/medium/large)
            output_format (str): 字幕输出格式 (txt/srt)
            include_timestamps (bool): 是否在输出中包含时间戳
            paragraphs (int): 每段的行数 (用于txt格式)
        """
        self.model_size = model_size
        self.output_format = output_format
        self.include_timestamps = include_timestamps
        self.paragraphs = paragraphs
        self.whisper_model = None
        # 初始化繁简中文转换器
        self.cc = OpenCC('t2s') if OpenCC else None

    def load_model(self):
        """将指定的Whisper模型加载到内存中
        
        返回:
            bool: 模型成功加载返回True，否则返回False
        """
        try:
            print(f"正在加载模型: {self.model_size}")
            # 加载模型配置，优先处理简体中文
            self.whisper_model = whisper.load_model(self.model_size)
            print(f"模型加载成功: {self.model_size}")
            return True
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False

    def transcribe_audio(self, audio_path):
        """使用Whisper模型转录给定的音频文件
        
        参数:
            audio_path (str): 要转录的音频文件路径
            
        返回:
            dict: 包含文本和时间戳的转录结果
        """
        if self.whisper_model is None:
            if not self.load_model():
                return None

        try:
            print(f"正在转录音频...")
            # 明确指定简体中文，避免繁体中文被误分类
            result = self.whisper_model.transcribe(
                audio_path,
                task="transcribe",  # 任务类型为转录
                language="chinese",  # 明确指定为中文
                fp16=torch.cuda.is_available()  # 使用GPU加速(如果可用)
            )
            # 验证检测到的语言
            detected_lang = result.get('language', 'unknown')
            print(f"检测到的语言: {detected_lang}")

            # 如果检测到繁体中文，强制转换为简体(备用处理)
            if detected_lang.lower().startswith('chinese (traditional)') and self.cc:
                for seg in result['segments']:
                    seg['text'] = self.cc.convert(seg['text'])
            return result
        except Exception as e:
            print(f"音频转录失败 - {audio_path}: {e}")
            return None

    def format_timestamp(self, seconds: float, always_include_hours: bool = False):
        """将时间戳格式化为SRT格式 (HH:MM:SS,mmm)
        
        参数:
            seconds (float): 以秒为单位的时间戳
            always_include_hours (bool): 是否始终在输出中包含小时
            
        返回:
            str: 格式化的时间戳字符串
        """
        assert seconds >= 0, f"需要非负时间戳: {seconds}"
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        # 格式化小时部分(如果有小时或强制包含小时)
        hours_marker = f"{int(hours):02d}:" if always_include_hours or hours > 0 else ""
        return f"{hours_marker}{int(minutes):02d}:{int(seconds):02d},{int((seconds % 1) * 1000):03d}"

    def extract_audio_from_video(self, video_path):
        """使用ffmpeg从视频文件中提取音频
        
        参数:
            video_path (str): 视频文件路径
            
        返回:
            str: 提取的音频文件路径，如果提取失败则返回None
        """
        audio_path = str(Path(video_path).with_suffix(".wav"))
        if not os.path.exists(audio_path):
            print(f"正在从视频中提取音频: {video_path}")
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-i", video_path, "-vn",  # -vn禁用视频处理
                        "-acodec", "pcm_s16le",  # 音频编码格式
                        "-ar", "16000",  # 采样率16kHz
                        "-ac", "1",  # 单声道
                        audio_path  # 输出音频路径
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,  # 隐藏标准输出
                    stderr=subprocess.DEVNULL  # 隐藏错误输出
                )
            except subprocess.CalledProcessError as e:
                print(f"音频提取失败: {e}")
                return None
        return audio_path

    def save_transcript(self, result, output_path):
        """将转录结果以指定格式保存到文件
        
        参数:
            result (dict): 包含文本和时间戳的转录结果
            output_path (str): 输出文件路径
            
        返回:
            bool: 保存成功返回True，否则返回False
        """
        try:
            if self.output_format.lower() == "txt":
                # 保存为TXT格式
                with open(output_path, "w", encoding="utf-8") as f:
                    segments = result["segments"]
                    for i, segment in enumerate(segments):
                        if self.include_timestamps:
                            start_time = self.format_timestamp(segment["start"])
                            f.write(f"[{start_time}] ")
                        # 再次转换以确保为简体中文(防止模型误分类)
                        text = segment['text'].strip()
                        if self.cc:
                            text = self.cc.convert(text)
                        f.write(f"{text}\n")
                        # 每paragraphs行添加一个空行，形成段落
                        if (i + 1) % self.paragraphs == 0:
                            f.write("\n")
                    f.write("\n")
            elif self.output_format.lower() == "srt":
                # 保存为SRT格式
                with open(output_path, "w", encoding="utf-8") as f:
                    segments = result["segments"]
                    for i, segment in enumerate(segments, start=1):
                        f.write(f"{i}\n")  # SRT序号
                        start_time = self.format_timestamp(segment["start"])
                        end_time = self.format_timestamp(segment["end"])
                        f.write(f"{start_time} --> {end_time}\n")  # 时间范围
                        text = segment['text'].strip()
                        if self.cc:
                            text = self.cc.convert(text)
                        f.write(f"{text}\n\n")  # 字幕文本
            else:
                print(f"不支持的输出格式: {self.output_format}")
                return False
            print(f"字幕已保存: {output_path}")
            return True
        except Exception as e:
            print(f"字幕保存失败 - {output_path}: {e}")
            return False

    def process_audio(self, audio_path):
        """处理单个音频或视频文件
        
        参数:
            audio_path (str): 音频或视频文件路径
            
        返回:
            bool: 处理成功返回True，否则返回False
        """
        media_suffix = Path(audio_path).suffix.lower()
        # 如果是视频文件，先提取音频
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

            # 检查文件是否已处理过
            if os.path.exists(output_path):
                print(f"跳过已处理的文件: {output_path}")
                return True

            print(f"\n正在处理音频: {audio_path}")
            result = self.transcribe_audio(audio_path)
            if result is None:
                return False
            return self.save_transcript(result, output_path)
        except Exception as e:
            print(f"音频处理失败 - {audio_path}: {e}")
            return False


def process_files(files, model_size="medium", output_format="txt", 
                  include_timestamps=True, paragraphs=10):
    """处理文件列表中的所有文件
    
    参数:
        files (list): 文件路径列表
        model_size (str): Whisper模型大小
        output_format (str): 输出格式
        include_timestamps (bool): 是否包含时间戳
        paragraphs (int): 每段行数
    """
    extractor = AudioSubtitleExtractor(
        model_size=model_size,
        output_format=output_format,
        include_timestamps=include_timestamps,
        paragraphs=paragraphs
    )
    success_count = 0
    total_files = len(files)

    for i, audio_path in enumerate(files, 1):
        print(f"正在处理文件 {i}/{total_files}: {os.path.basename(audio_path)}")
        if extractor.process_audio(audio_path):
            success_count += 1

    print(f"处理完成: 成功={success_count}, 失败={total_files - success_count}, 总计={total_files}")


def get_files_from_path(path):
    """从给定路径获取所有媒体文件
    
    参数:
        path (str): 文件或目录路径
        
    返回:
        list: 媒体文件路径列表
    """
    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        # 支持的音频和视频格式
        media_extensions = [
            '.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac',  # 音频格式
            '.mp4', '.mkv', '.avi', '.mov', '.webm'  # 视频格式
        ]
        return [
            os.path.join(path, f) for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
               and Path(f).suffix.lower() in media_extensions
        ]
    else:
        print(f"无效路径: {path}")
        return []


if __name__ == "__main__":
    """程序入口点"""
    if len(sys.argv) < 2:
        print("使用方法: python script.py <文件或目录路径> [模型大小] [输出格式] [是否包含时间戳] [段落行数]")
        sys.exit(1)

    path = sys.argv[1]  # 必需的文件或目录路径
    model_size = sys.argv[2] if len(sys.argv) > 2 else "medium"  # 默认为medium模型
    output_format = sys.argv[3] if len(sys.argv) > 3 else "txt"  # 默认为txt格式
    include_timestamps = bool(int(sys.argv[4])) if len(sys.argv) > 4 else True  # 默认包含时间戳
    paragraphs = int(sys.argv[5]) if len(sys.argv) > 5 else 10  # 默认每10行一段

    files = get_files_from_path(path)
    if files:
        process_files(files, model_size, output_format, include_timestamps, paragraphs)
