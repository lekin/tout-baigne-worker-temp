"""Core karaoke generation functionality."""
import os
import re
import subprocess
import tempfile
import shutil
import ssl
import certifi
import json
import sys
import random
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import yt_dlp
from demucs.pretrained import get_model
from demucs.apply import apply_model
import torch
import torchaudio
from src.config import settings

# Fix SSL certificate issues
import urllib.request
ssl._create_default_https_context = ssl._create_unverified_context


def _demucs_device() -> torch.device:
    """Return the best available torch device for Demucs separation."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class KaraokeGenerator:
    """Handles karaoke video generation from music videos."""
    
    def __init__(
        self,
        video_width: Optional[int] = None,
        video_height: Optional[int] = None,
        logo_width: Optional[int] = None,
        logo_height: Optional[int] = None,
        font_path: Optional[str] = None,
        logo_path: Optional[str] = None,
        whisper_model: Optional[str] = None
    ):
        """
        Initialize the karaoke generator.
        
        Args:
            video_width: Output video width
            video_height: Output video height
            logo_width: Logo width
            logo_height: Logo height
            font_path: Path to font file
            logo_path: Path to logo file
            whisper_model: Whisper model size (tiny, base, small, medium, large)
        """
        self.video_width = video_width or settings.karaoke_video_width
        self.video_height = video_height or settings.karaoke_video_height
        self.logo_width = logo_width or settings.karaoke_logo_width
        self.logo_height = logo_height or settings.karaoke_logo_height
        self.font_path = font_path or settings.karaoke_font_path
        self.logo_path = logo_path or settings.karaoke_logo_path
        self.whisper_model_size = whisper_model or "large-v3"  # Default to large-v3 for best quality
        self._whisper_model = None
        self._last_effect_path: Optional[str] = None
    
    def get_whisper_model(self):
        """Legacy stub: Whisper-based transcription has been removed.

        This project now relies on Musixmatch lyrics (SRT / Richsync) instead of
        on-device transcription. This method is kept only to avoid hard
        dependency on faster-whisper. Any call will raise a RuntimeError.
        """
        raise RuntimeError(
            "Whisper-based transcription is disabled in this project. "
            "Use Musixmatch-provided lyrics instead."
        )
    
    def _transcribe_with_whisperx(
        self,
        audio_path: str,
        language: Optional[str] = None,
        genius_lyrics: Optional[str] = None
    ):
        """Legacy stub kept for backwards compatibility.

        WhisperX alignment has been removed from this project. All
        transcription now relies on external Musixmatch lyrics. Any attempt
        to call this method will raise a RuntimeError to make this explicit.
        """
        raise RuntimeError(
            "WhisperX-based transcription has been removed. "
            "Use Musixmatch lyrics instead."
        )
    
    def _transcribe_with_faster_whisper_only(
        self,
        audio_path: str,
        language: Optional[str] = None,
        genius_lyrics: Optional[str] = None,
        audio_duration: Optional[float] = None
    ) -> Tuple[List[Dict], str, float]:
        """Legacy stub: faster-whisper transcription removed.

        The project no longer performs on-device transcription and instead uses
        Musixmatch lyrics. This method is retained only to avoid breaking
        imports; calling it will raise a RuntimeError.
        """
        raise RuntimeError(
            "faster-whisper transcription has been removed from this project. "
            "Use Musixmatch-based workflows instead."
        )
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds to SRT timestamp (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def _segments_to_word_based_srt(self, segments: List[dict], max_words_per_line: int = 8) -> str:
        """
        Convert segments to SRT with word-level splitting for better precision.
        Creates shorter subtitle lines based on word timestamps.
        
        Args:
            segments: List of segment dicts with 'words' arrays
            max_words_per_line: Maximum words per subtitle line (default: 8)
        
        Returns:
            SRT formatted string with shorter, more precise lines
        """
        srt_lines = []
        line_num = 1
        
        for segment in segments:
            words = segment.get('words', [])
            
            if not words:
                # Fallback: use segment-level timestamps if no words
                start_time = self._format_timestamp(segment['start'])
                end_time = self._format_timestamp(segment['end'])
                text = segment['text'].strip()
                
                srt_lines.append(f"{line_num}")
                srt_lines.append(f"{start_time} --> {end_time}")
                srt_lines.append(text)
                srt_lines.append("")
                line_num += 1
                continue
            
            # Split words into chunks
            i = 0
            while i < len(words):
                # Take up to max_words_per_line words
                chunk = words[i:i + max_words_per_line]
                
                # Get timestamps from first and last word in chunk
                start_time = self._format_timestamp(chunk[0]['start'])
                end_time = self._format_timestamp(chunk[-1]['end'])
                
                # Combine words into text
                text = ' '.join(str(w.get('word', '')).strip() for w in chunk).strip()
                
                if text:  # Only add if there's actual text
                    srt_lines.append(f"{line_num}")
                    srt_lines.append(f"{start_time} --> {end_time}")
                    srt_lines.append(text)
                    srt_lines.append("")
                    line_num += 1
                
                i += max_words_per_line
        
        return "\n".join(srt_lines)
    
    def _whisperx_to_srt(self, segments: List[dict]) -> str:
        """Convert WhisperX segments to SRT format."""
        srt_lines = []
        for i, segment in enumerate(segments, 1):
            start_time = self._format_timestamp(segment['start'])
            end_time = self._format_timestamp(segment['end'])
            text = segment['text'].strip()
            
            srt_lines.append(f"{i}")
            srt_lines.append(f"{start_time} --> {end_time}")
            srt_lines.append(text)
            srt_lines.append("")  # Empty line between subtitles
        
        return "\n".join(srt_lines)
    
    def _analyze_lyrics_and_get_vad_params(self, genius_lyrics: Optional[str] = None) -> dict:
        """
        Analyze Genius lyrics to determine optimal Whisper VAD parameters.
        
        Returns:
            dict: VAD parameters optimized for the song type
        """
        if not genius_lyrics:
            # Default balanced parameters
            return dict(
                min_silence_duration_ms=400,
                threshold=0.5,
                min_speech_duration_ms=250
            )
        
        lines = [l.strip() for l in genius_lyrics.split('\n') if l.strip()]
        
        if not lines:
            return dict(
                min_silence_duration_ms=400,
                threshold=0.5,
                min_speech_duration_ms=250
            )
        
        # Analyze lyrics characteristics
        total_lines = len(lines)
        short_lines = sum(1 for l in lines if len(l) < 20)  # Ad-libs, hooks
        very_short_lines = sum(1 for l in lines if len(l) < 10)  # "Yeah", "Ahh", etc.
        avg_line_length = sum(len(l) for l in lines) / total_lines
        
        # Count repetitions (chorus detection)
        from collections import Counter
        line_counts = Counter(lines)
        repeated_lines = sum(1 for count in line_counts.values() if count > 1)
        repetition_ratio = repeated_lines / total_lines
        
        # Determine song type and adapt parameters
        short_line_ratio = short_lines / total_lines
        very_short_ratio = very_short_lines / total_lines
        
        print(f"📊 Lyrics analysis: {total_lines} lines, avg length: {avg_line_length:.1f}, short: {short_line_ratio:.1%}, repetitions: {repetition_ratio:.1%}")
        
        # Adaptive parameters based on analysis
        if very_short_ratio > 0.3:
            # Many ad-libs/hooks (like "Yeah", "Ahh")
            print("   🎵 Detected: High ad-lib content → Aggressive capture mode")
            return dict(
                min_silence_duration_ms=200,  # Very short pauses
                threshold=0.3,  # Lower threshold to catch quiet ad-libs
                min_speech_duration_ms=100  # Catch very short utterances
            )
        elif short_line_ratio > 0.5:
            # Many short lines (hooks, chorus)
            print("   🎵 Detected: Short lines/hooks → Enhanced sensitivity")
            return dict(
                min_silence_duration_ms=300,
                threshold=0.4,
                min_speech_duration_ms=150
            )
        elif repetition_ratio > 0.5:
            # Highly repetitive (chorus-heavy)
            print("   🎵 Detected: Repetitive/chorus-heavy → Balanced mode")
            return dict(
                min_silence_duration_ms=350,
                threshold=0.45,
                min_speech_duration_ms=200
            )
        else:
            # Standard song (verses, narrative)
            # ULTRA PERMISSIVE MODE - capture EVERYTHING from separated vocals
            print("   🎵 Detected: Standard song structure → ULTRA PERMISSIVE mode")
            return dict(
                min_silence_duration_ms=150,  # Very short pauses
                threshold=0.2,  # Very low threshold - catch everything
                min_speech_duration_ms=50  # Catch even the shortest sounds
            )
    
    @staticmethod
    def download_youtube_video(
        url: str,
        output_path: str,
        use_cookies: bool = True,
        max_height: int = 1080,
        cookies_txt_path: Optional[str] = None,
    ) -> bool:
        """
        Download a video from YouTube.
        
        Args:
            url: YouTube video URL
            output_path: Path to save the video
            use_cookies: If True, try to use browser cookies to bypass bot detection
            
        Returns:
            True if successful, False otherwise
        """
        try:

            def _valid_file() -> bool:
                if not os.path.exists(output_path):
                    return False
                try:
                    return os.path.getsize(output_path) > 1000
                except Exception:
                    return False

            def _cleanup_empty_output() -> None:
                try:
                    if os.path.exists(output_path) and os.path.getsize(output_path) <= 1000:
                        os.unlink(output_path)
                except Exception:
                    pass

                try:
                    alt = output_path + '.part.mp4'
                    if os.path.exists(alt) and os.path.getsize(alt) <= 1000:
                        os.unlink(alt)
                except Exception:
                    pass

            def _finalize_cli_output() -> bool:
                if _valid_file():
                    return True

                alt_candidates = [
                    output_path + '.part.mp4',
                    output_path + '.part',
                ]
                for alt in alt_candidates:
                    try:
                        if os.path.exists(alt) and os.path.getsize(alt) > 1000:
                            try:
                                os.replace(alt, output_path)
                            except Exception:
                                try:
                                    os.rename(alt, output_path)
                                except Exception:
                                    continue
                            return _valid_file()
                    except Exception:
                        continue
                return _valid_file()

            def _try_cli_download() -> bool:
                if not (cookies_txt_path and isinstance(cookies_txt_path, str) and os.path.exists(cookies_txt_path)):
                    return False

                node_path = shutil.which('node')
                if not node_path:
                    return False

                fmt = (
                    f"bestvideo[height<={mh}][vcodec^=avc][ext=mp4]"
                    f"+bestaudio[ext=m4a]"
                    f"/bestvideo[height<={mh}][ext=mp4]+bestaudio"
                    f"/best[height<={mh}][ext=mp4]"
                    f"/best[ext=mp4]"
                    f"/best"
                )

                if isinstance(output_path, str) and output_path.lower().endswith('.mp4'):
                    outtmpl = output_path[:-4] + '.%(ext)s'
                else:
                    outtmpl = output_path + '.%(ext)s'

                cmd = [
                    sys.executable,
                    '-m',
                    'yt_dlp',
                    '--no-playlist',
                    '--force-overwrites',
                    '--merge-output-format',
                    'mp4',
                    '--output',
                    outtmpl,
                    '--cookies',
                    cookies_txt_path,
                    '--impersonate',
                    'chrome',
                    '--no-js-runtimes',
                    '--js-runtimes',
                    'node',
                    '--remote-components',
                    'ejs:github',
                    '--format-sort',
                    f"res:{mh},ext:mp4:m4a",
                    '-f',
                    fmt,
                    url,
                ]

                print("Attempting download via yt-dlp CLI (impersonate+node+EJS)...")
                _cleanup_empty_output()
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        print(line.rstrip())
                    rc = proc.wait()
                    if rc == 0 and _finalize_cli_output():
                        file_size = os.path.getsize(output_path)
                        print(f"✓ Downloaded {file_size / (1024*1024):.1f} MB")
                        return True
                    _finalize_cli_output()
                    return False
                except Exception:
                    return False
                finally:
                    _cleanup_empty_output()

            base_opts = {
                'outtmpl': output_path,
                'quiet': False,
                'no_warnings': False,
                'overwrites': True,
                'nocheckcertificate': True,
                'ignoreerrors': False,
                'no_color': True,
                'noplaylist': True,
                'retries': 3,
                'fragment_retries': 5,
                'http_headers': {
                    'User-Agent': (
                        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                        'AppleWebKit/537.36 (KHTML, like Gecko) '
                        'Chrome/122.0.0.0 Safari/537.36'
                    ),
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Referer': 'https://www.youtube.com/',
                    'Origin': 'https://www.youtube.com',
                },
            }

            if cookies_txt_path:
                try:
                    if isinstance(cookies_txt_path, str) and os.path.exists(cookies_txt_path):
                        base_opts['cookiefile'] = cookies_txt_path
                except Exception:
                    pass

            try:
                mh = int(max_height)
            except Exception:
                mh = 1080
            if mh <= 0:
                mh = 1080

            if cookies_txt_path:
                return _try_cli_download()

            if base_opts.get('cookiefile'):
                print(f"Attempting download with cookies file: {base_opts.get('cookiefile')}")
            else:
                print("Attempting download without browser cookies...")

            attempt_opts = [
                {
                    **base_opts,
                    'format': 'best[protocol^=https][ext=mp4]/best[ext=mp4]/best',
                    'merge_output_format': 'mp4',
                    'extractor_args': {
                        'youtube': {
                            'player_client': ['web_safari', 'web'],
                        }
                    },
                },
                {
                    **base_opts,
                    'format': (
                        f"bestvideo[height<={mh}][protocol^=https][vcodec^=avc][ext=mp4]"
                        f"+bestaudio[protocol^=https][ext=m4a]"
                        f"/best[height<={mh}][protocol^=https][ext=mp4]"
                        f"/best[protocol^=https][ext=mp4]"
                        f"/best"
                    ),
                    'merge_output_format': 'mp4',
                    'extractor_args': {
                        'youtube': {
                            'player_client': ['web_safari', 'web'],
                        }
                    },
                },
                {
                    **base_opts,
                    'retries': 2,
                    'fragment_retries': 2,
                    'format': f'bestvideo[height<={mh}][vcodec^=avc]+bestaudio/best[height<={mh}]/best',
                    'merge_output_format': 'mp4',
                    'hls_prefer_native': True,
                    'extractor_args': {
                        'youtube': {
                            'player_client': ['android', 'web_safari', 'web'],
                        }
                    },
                },
            ]

            last_error: Optional[Exception] = None
            seen_errors: list[str] = []
            drm_detected = False
            for opts in attempt_opts:
                _cleanup_empty_output()
                try:
                    with yt_dlp.YoutubeDL(opts) as ydl:
                        ydl.extract_info(url, download=True)
                    if _valid_file():
                        file_size = os.path.getsize(output_path)
                        print(f"✓ Downloaded {file_size / (1024*1024):.1f} MB")
                        return True
                except Exception as e:
                    last_error = e
                    err_msg = str(e).lower()
                    if 'drm protected' in err_msg:
                        drm_detected = True
                        break
                    try:
                        seen_errors.append(err_msg)
                    except Exception:
                        pass

                    continue
                finally:
                    _cleanup_empty_output()

            if drm_detected:
                print("⚠️  YouTube reported this video as DRM protected; skipping download and using fallback background")
                return False

            # Retry once with Chrome cookies only if it looks like authentication is required.
            if use_cookies and (not cookies_txt_path) and last_error is not None:
                msg = str(last_error).lower()
                combined = " ".join([msg] + seen_errors)
                auth_markers = (
                    'sign in',
                    'login',
                    'bot',
                    'http error 403',
                    '403',
                    'forbidden',
                    'po token',
                    'gvs',
                    'confirm your age',
                    'age-restricted',
                    'members-only',
                    'private',
                )
                if any(m in combined for m in auth_markers):
                    print("⚠️  Retrying with chrome cookies (may prompt for Keychain access)...")
                    cookies_base = {
                        **base_opts,
                        'cookiesfrombrowser': ('chrome',),
                    }
                    cookies_attempts = []
                    for o in attempt_opts:
                        oo = {**o}
                        oo.update(cookies_base)
                        # yt-dlp warns that android client does not support cookies.
                        extractor_args = oo.get('extractor_args') or {}
                        yt_args = extractor_args.get('youtube') or {}
                        yt_args = {**yt_args, 'player_client': ['web_safari', 'web']}
                        extractor_args = {**extractor_args, 'youtube': yt_args}
                        oo['extractor_args'] = extractor_args
                        cookies_attempts.append(oo)

                    for opts in cookies_attempts:
                        _cleanup_empty_output()
                        try:
                            with yt_dlp.YoutubeDL(opts) as ydl:
                                ydl.extract_info(url, download=True)
                            if _valid_file():
                                file_size = os.path.getsize(output_path)
                                print(f"✓ Downloaded {file_size / (1024*1024):.1f} MB")
                                return True
                        except Exception as e:
                            last_error = e
                            continue
                        finally:
                            _cleanup_empty_output()

            if last_error is not None:
                raise last_error

            print("Error: Download failed")
            return False
                
        except Exception as e:
            print(f"Error downloading video: {e}")
            import traceback
            traceback.print_exc()
            return False

    def render_sync_preview(self, audio_path: str, ass_path: str, output_path: str, fast_mode: bool = True, bg_color_hex: str = "#000000", width: Optional[int] = None, height: Optional[int] = None, fps: Optional[int] = None, video_bitrate: str = "2000k", audio_bitrate: str = "192k") -> Tuple[bool, Optional[str]]:
        try:
            w = int(width or self.video_width)
            h = int(height or self.video_height)

            if not os.path.exists(audio_path):
                return False, f"Audio file not found: {audio_path}"
            if not os.path.exists(ass_path):
                return False, f"ASS file not found: {ass_path}"

            ass_path_escaped = (ass_path
                .replace('\\', '\\\\')
                .replace(':', '\\:')
                .replace("'", "\\'")
                .replace(',', '\\,')
                .replace('[', '\\[')
                .replace(']', '\\]')
            )

            out_fps = int(fps or (24 if fast_mode else 30))

            cmd = [
                'ffmpeg', '-y',
                '-f', 'lavfi',
                '-i', f"color=c={bg_color_hex}:s={w}x{h}:r={out_fps}",
                '-i', audio_path,
                '-vf', f"ass={ass_path_escaped}",
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac', '-b:a', str(audio_bitrate),
                '-movflags', '+faststart',
                '-shortest'
            ]

            if fast_mode:
                cmd.extend(['-c:v', 'h264_videotoolbox', '-b:v', str(video_bitrate), '-r', str(out_fps)])
            else:
                cmd.extend(['-c:v', 'libx264', '-crf', '23', '-preset', 'ultrafast', '-r', str(out_fps)])

            cmd.append(output_path)

            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if result.returncode != 0:
                return False, result.stderr

            return True, output_path
        except Exception as e:
            return False, str(e)
    
    @staticmethod
    def separate_vocals_ffmpeg(audio_path: str, output_dir: Optional[str] = None, keep_files: bool = False) -> Optional[str]:
        """
        Fast vocal isolation using FFmpeg audio filters (center channel extraction).
        Much faster than AI models but lower quality.
        
        Args:
            audio_path: Path to audio/video file
            output_dir: Optional output directory
            keep_files: If True, saves vocals to output directory
            
        Returns:
            Path to isolated vocals file or None if failed
        """
        try:
            print("🎵 Isolating vocals with FFmpeg (fast method)...")
            print("⏱️  This will take 10-30 seconds...")
            
            # Create output directory
            if output_dir is None:
                if keep_files:
                    output_dir = os.path.join(settings.output_dir, "separated_vocals")
                    os.makedirs(output_dir, exist_ok=True)
                else:
                    output_dir = tempfile.mkdtemp()
            
            # Generate output filename
            if keep_files:
                video_name = os.path.splitext(os.path.basename(audio_path))[0]
                vocals_path = os.path.join(output_dir, f"{video_name}_vocals.wav")
            else:
                vocals_path = os.path.join(output_dir, "vocals.wav")
            
            # Use FFmpeg to isolate center channel (where vocals usually are)
            # This extracts vocals by removing side channels
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-af", "pan=stereo|c0=c0-c1|c1=c0-c1,volume=2",  # Center channel extraction
                "-ar", "44100",
                "-ac", "2",
                vocals_path
            ]
            
            print("📼 Processing audio...")
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            
            if result.returncode != 0:
                print(f"Error isolating vocals: {result.stderr}")
                return None
            
            # Verify the file was created
            if not os.path.exists(vocals_path):
                print("❌ Error: Vocals file was not created")
                return None
            
            file_size = os.path.getsize(vocals_path)
            if file_size < 1000:
                print(f"❌ Error: Vocals file is too small ({file_size} bytes)")
                return None
            
            print(f"✓ Vocals isolated and saved: {vocals_path} ({file_size / (1024*1024):.1f} MB)")
            return vocals_path
            
        except Exception as e:
            print(f"Error isolating vocals: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def separate_vocals(audio_path: str, output_dir: Optional[str] = None, keep_files: bool = False) -> Optional[str]:
        """
        Separate vocals from music using Demucs AI.
        
        Args:
            audio_path: Path to audio/video file
            output_dir: Optional output directory (uses temp if not provided)
            keep_files: If True, saves vocals to output directory instead of temp
            
        Returns:
            Path to separated vocals file or None if failed
        """
        try:
            print("🎵 Separating vocals with Demucs AI...")
            print("⏱️  This will take 5-10 minutes on CPU. Please be patient...")
            
            # Create output directory
            if output_dir is None:
                if keep_files:
                    output_dir = os.path.join(settings.output_dir, "separated_vocals")
                    os.makedirs(output_dir, exist_ok=True)
                else:
                    output_dir = tempfile.mkdtemp()
            
            # Load Demucs model (htdemucs is the best quality)
            model = get_model('htdemucs')

            # Use GPU (Metal Performance Shaders on Apple Silicon, CUDA otherwise)
            # when available to speed up separation significantly.
            device = _demucs_device()
            model.to(device)
            model.eval()
            
            # Extract audio from video if needed (MP4 -> WAV)
            audio_file = audio_path
            temp_audio = None
            
            if audio_path.lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):
                print("📼 Extracting audio from video...")
                temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_audio.close()
                
                # Use FFmpeg to extract audio
                extract_cmd = [
                    "ffmpeg", "-y", "-i", audio_path,
                    "-vn",  # No video
                    "-acodec", "pcm_s16le",  # PCM audio
                    "-ar", "44100",  # 44.1kHz
                    "-ac", "2",  # Stereo
                    temp_audio.name
                ]
                
                result = subprocess.run(extract_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
                if result.returncode != 0:
                    print(f"Error extracting audio: {result.stderr}")
                    if os.path.exists(temp_audio.name):
                        os.unlink(temp_audio.name)
                    return None
                
                audio_file = temp_audio.name
                print(f"✓ Audio extracted successfully")
            
            # Load audio
            try:
                wav, sr = torchaudio.load(audio_file)
            except Exception as e:
                print(f"Error loading audio: {e}")
                if temp_audio and os.path.exists(temp_audio.name):
                    os.unlink(temp_audio.name)
                return None
            
            # Ensure stereo
            if wav.shape[0] == 1:
                wav = wav.repeat(2, 1)
            
            # Resample if needed (Demucs expects 44.1kHz)
            if sr != 44100:
                resampler = torchaudio.transforms.Resample(sr, 44100)
                wav = resampler(wav)
                sr = 44100
            
            # Apply model with progress indication.
            # Use shifts=0 for speed; the default shifts=1 doubles compute and
            # the gain in quality is not needed for QA-level separation.
            device_name = "MPS" if device.type == "mps" else ("CUDA" if device.type == "cuda" else "CPU")
            print(f"Processing audio with Demucs on {device_name} (shifts=0)...")
            print("Progress: Starting separation...")

            with torch.no_grad():
                wav_input = wav.unsqueeze(0).to(device)
                try:
                    sources = apply_model(
                        model, wav_input, device=device, shifts=0
                    )[0]
                except (NotImplementedError, RuntimeError) as e:
                    if device.type != "cpu":
                        print(f"⚠️ {device.type.upper()} failed ({e}), falling back to CPU...")
                        device = torch.device("cpu")
                        model.to(device)
                        wav_input = wav_input.cpu()
                        sources = apply_model(
                            model, wav_input, device=device, shifts=0
                        )[0]
                    else:
                        raise
            
            print("Progress: Separation complete, extracting vocals...")
            
            # Extract vocals (index 3 in htdemucs: drums, bass, other, vocals)
            vocals = sources[3]
            
            # Save vocals
            if keep_files:
                # Use descriptive filename
                video_name = os.path.splitext(os.path.basename(audio_path))[0]
                vocals_path = os.path.join(output_dir, f"{video_name}_vocals.wav")
            else:
                vocals_path = os.path.join(output_dir, "vocals.wav")
            
            torchaudio.save(vocals_path, vocals.cpu(), sr)
            
            # Verify the file was created and has content
            if not os.path.exists(vocals_path):
                print("❌ Error: Vocals file was not created")
                return None
            
            file_size = os.path.getsize(vocals_path)
            if file_size < 1000:
                print(f"❌ Error: Vocals file is too small ({file_size} bytes)")
                return None
            
            if keep_files:
                print(f"✓ Vocals separated and saved: {vocals_path} ({file_size / (1024*1024):.1f} MB)")
            else:
                print(f"✓ Vocals separated: {vocals_path} ({file_size / (1024*1024):.1f} MB)")
            
            # Clean up temporary audio file
            if temp_audio and os.path.exists(temp_audio.name):
                try:
                    os.unlink(temp_audio.name)
                except:
                    pass
            
            return vocals_path
            
        except Exception as e:
            print(f"Error separating vocals: {e}")
            import traceback
            traceback.print_exc()
            
            # Clean up temporary audio file on error
            if temp_audio and os.path.exists(temp_audio.name):
                try:
                    os.unlink(temp_audio.name)
                except:
                    pass
            
            return None
    
    def transcribe_video(self, video_path: str, separate_vocals: bool = False, genius_lyrics: Optional[str] = None, preserve_timecodes: bool = False, language: Optional[str] = None) -> Optional[str]:
        """
        Transcribe video audio to SRT format using Whisper.
        
        Args:
            video_path: Path to video file
            separate_vocals: If True, separate vocals first for better accuracy
            
        Returns:
            SRT content string or None if failed
        """
        vocals_path = None
        temp_dir = None
        
        try:
            # Separate vocals if requested (ALWAYS use Lalal.ai - no FFmpeg fallback)
            if separate_vocals:
                print("🎵 Starting Lalal.ai vocal separation...")
                try:
                    from src.lalal_api import LalalAIClient
                    lalal = LalalAIClient()
                    
                    # Create output directory
                    vocals_dir = os.path.join(settings.output_dir, "separated_vocals")
                    os.makedirs(vocals_dir, exist_ok=True)
                    
                    # Generate output filename
                    video_name = os.path.splitext(os.path.basename(video_path))[0]
                    
                    # First, extract audio from video (Lalal.ai works better with audio files)
                    print("🎧 Extracting audio from video...")
                    temp_audio = os.path.join(vocals_dir, f"{video_name}_temp_audio.mp3")
                    audio_cmd = [
                        "ffmpeg", "-y", "-i", video_path,
                        "-vn",  # No video
                        "-acodec", "libmp3lame",
                        "-b:a", "192k",
                        temp_audio
                    ]
                    subprocess.run(audio_cmd, check=True, capture_output=True)
                    print(f"✓ Audio extracted: {os.path.getsize(temp_audio) / (1024*1024):.1f} MB")
                    
                    vocals_path = os.path.join(vocals_dir, f"{video_name}_lalal_vocals.mp3")
                    
                    # Separate with Lalal.ai using audio file
                    vocals_path = lalal.separate_vocals(temp_audio, vocals_path)
                    
                    # Clean up temp audio
                    try:
                        os.unlink(temp_audio)
                    except:
                        pass
                    
                    # Upload vocal to Airtable automatically
                    if vocals_path and hasattr(self, '_airtable_record_id') and self._airtable_record_id:
                        print(f"💾 Vocal file ready: {os.path.basename(vocals_path)} ({os.path.getsize(vocals_path)/(1024*1024):.1f} MB)")
                        
                        from src.airtable_client import AirtableClient
                        airtable = AirtableClient()
                        if airtable.upload_vocal_file(vocals_path, self._airtable_record_id):
                            print(f"✅ Vocal uploaded to Airtable")
                        else:
                            print(f"⚠️  Vocal upload failed, you may need to upload manually")
                    
                    if not vocals_path:
                        print("❌ Lalal.ai separation failed!")
                        print("⚠️  Stopping process - vocal separation is required")
                        return None
                    
                    print("✅ Vocal separation successful! Using isolated vocals for transcription.")
                    audio_to_transcribe = vocals_path
                    temp_dir = None  # Don't clean up when keeping files
                    
                except Exception as e:
                    print(f"❌ Lalal.ai error: {e}")
                    print("⚠️  Stopping process - vocal separation is required")
                    import traceback
                    traceback.print_exc()
                    return None
            else:
                # No separation needed - use the audio file directly
                # This happens when vocals_file is already provided or when using original audio
                print("🎵 Using audio file directly (no separation needed)")
                audio_to_transcribe = video_path  # This is actually the audio/vocal file path
                temp_dir = None

            # Use provided language or auto-detect
            transcribe_language = language if language else None  # None = auto-detect
            if transcribe_language:
                print(f"🌍 Using language: {transcribe_language}")
            else:
                print("🌍 Auto-detecting language...")

            # Get audio duration for section-based transcription
            try:
                audio_probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", audio_to_transcribe],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=True
                )
                audio_duration = float(audio_probe.stdout.strip())
            except Exception as e:
                print(f"⚠️ Could not get audio duration: {e}")
                audio_duration = None

            segments_list, detected_lang, detected_prob = self._transcribe_with_faster_whisper_only(
                audio_to_transcribe,
                language=transcribe_language,
                genius_lyrics=genius_lyrics,
                audio_duration=audio_duration
            )

            print(f"Detected language: {detected_lang} (probability: {detected_prob:.2f})")

            # Extract raw transcription text (without timestamps)
            raw_transcription = ' '.join(seg['text'].strip() for seg in segments_list if seg['text'].strip())

            # Create transcription with timestamps (human-readable format)
            transcription_with_timestamps = []
            for seg in segments_list:
                if seg['text'].strip():
                    start_time = self._format_timestamp(seg['start'])
                    end_time = self._format_timestamp(seg['end'])
                    transcription_with_timestamps.append(f"[{start_time} --> {end_time}] {seg['text'].strip()}")
            transcription_with_timestamps_text = '\n'.join(transcription_with_timestamps)

            # Save detected language and raw transcription to Airtable if we have a record ID
            if hasattr(self, '_airtable_record_id') and self._airtable_record_id:
                try:
                    from src.airtable_client import AirtableClient
                    airtable = AirtableClient()

                    updates = {}

                    # Save language if not manually specified
                    if not language:
                        updates["Language"] = detected_lang

                    # Save raw transcription
                    updates["Whisper transcription"] = raw_transcription

                    # Save transcription with timestamps
                    updates["Whisper transcription (timestamps)"] = transcription_with_timestamps_text

                    if updates:
                        airtable.update_record(self._airtable_record_id, updates)
                        print(f"✓ Saved to Airtable: {', '.join(updates.keys())}")
                except Exception as e:
                    print(f"⚠️  Could not save to Airtable: {e}")

            # Convert to SRT format with word-level splitting for better precision
            # This creates shorter, more precise subtitle lines
            whisper_srt_content = self._segments_to_word_based_srt(segments_list)

            # Only normalize if NOT preserving original timecodes
            if not preserve_timecodes:
                whisper_srt_content = self.normalize_srt_to_zero(whisper_srt_content)

            # Save Whisper SRT to Airtable BEFORE Genius sync
            if hasattr(self, '_airtable_record_id') and self._airtable_record_id:
                try:
                    from src.airtable_client import AirtableClient
                    airtable = AirtableClient()
                    print("💾 Saving Whisper SRT to Airtable...")
                    airtable.save_srt_only(self._airtable_record_id, whisper_srt_content, srt_type="whisper")
                    print("✅ Whisper SRT saved to Airtable")
                except Exception as e:
                    print(f"⚠️  Could not save Whisper SRT to Airtable: {e}")

            # If we have Genius lyrics, sync them with Whisper timing
            if genius_lyrics:
                print("🎵 Syncing Genius lyrics with Whisper word-level timestamps...")
                from src.advanced_lyrics_aligner import AdvancedLyricsAligner

                # Extract word-level timestamps from segments
                whisper_words = []
                for seg in segments_list:
                    # segments_list is now a list of dicts (from faster-whisper)
                    if isinstance(seg, dict) and 'words' in seg and seg['words']:
                        for word in seg['words']:
                            whisper_words.append({
                                'word': word['word'],
                                'start': word['start'],
                                'end': word['end'],
                                'confidence': word.get('probability', 1.0)
                            })

                if whisper_words:
                    print(f"✓ Extracted {len(whisper_words)} word timestamps from Whisper")

                    # Save word timestamps to Airtable for future use
                    if hasattr(self, '_airtable_record_id') and self._airtable_record_id:
                        try:
                            from src.airtable_client import AirtableClient
                            airtable = AirtableClient()
                            words_json = json.dumps(whisper_words)
                            airtable.update_record(self._airtable_record_id, {"Whisper words (JSON)": words_json})
                            print(f"✓ Saved word timestamps to Airtable")
                        except Exception as e:
                            print(f"⚠️  Could not save word timestamps: {e}")

                    # Use advanced aligner
                    aligner = AdvancedLyricsAligner()
                    synced_srt, report = aligner.align_lyrics_to_whisper(genius_lyrics, whisper_words)

                    # Save report
                    if hasattr(self, '_airtable_record_id') and self._airtable_record_id:
                        report_path = video_path.replace('.mp4', '_alignment_report.json')
                        with open(report_path, 'w') as f:
                            json.dump(report, f, indent=2)
                        print(f"✓ Saved alignment report: {report_path}")
                        print(f"📊 Alignment confidence: {report['summary']['avg_confidence']:.1%}")
                        print(f"   High confidence lines: {report['summary']['high_confidence_lines']}/{report['summary']['total_lines']}")
                        if report['summary']['low_confidence_lines'] > 0:
                            print(f"   ⚠️  Low confidence lines: {report['summary']['low_confidence_lines']}")
                else:
                    print("⚠️  No word timestamps available, using fallback alignment")
                    from src.lyrics_sync import match_lyrics_to_segments

                    # segments_list is already in dict format from faster-whisper
                    synced_srt = match_lyrics_to_segments(genius_lyrics, segments_list)

                if synced_srt:
                    print("✓ Successfully synced Genius lyrics with Whisper timing!")

                    # Only normalize if NOT preserving original timecodes
                    if not preserve_timecodes:
                        synced_srt = self.normalize_srt_to_zero(synced_srt)
                        print("✓ Normalized SRT to start at 00:00:00")
                    else:
                        print("✓ Preserving original Whisper timecodes for Genius sync")

                    # Save synced SRT file
                    self._save_srt_file(video_path, synced_srt, "genius_synced")

                    # Save to Airtable as Genius SRT
                    if hasattr(self, '_airtable_record_id') and self._airtable_record_id:
                        try:
                            from src.airtable_client import AirtableClient
                            airtable = AirtableClient()
                            print("💾 Saving Genius SRT to Airtable...")
                            airtable.save_srt_only(self._airtable_record_id, synced_srt, srt_type="genius")
                            print("✅ Genius SRT saved to Airtable")
                        except Exception as e:
                            print(f"⚠️  Could not save Genius SRT to Airtable: {e}")

                    return synced_srt
                else:
                    print("⚠️  Sync failed, using Whisper transcription")

            # Return Whisper SRT (already saved above)
            return whisper_srt_content
            
        except Exception as e:
            print(f"Error transcribing video: {e}")
            return None
        finally:
            # Clean up temporary vocals file
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
    
    def _save_srt_file(self, video_path: str, srt_content: str, suffix: str) -> None:
        """
        Save SRT content to a file in the output directory.
        
        Args:
            video_path: Original video path (used for naming)
            srt_content: SRT content to save
            suffix: Suffix to add to filename (e.g., 'whisper', 'genius_synced')
        """
        try:
            # Create SRT output directory
            srt_dir = os.path.join(settings.output_dir, "srt_files")
            os.makedirs(srt_dir, exist_ok=True)
            
            # Generate filename
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            srt_filename = f"{video_name}_{suffix}.srt"
            srt_path = os.path.join(srt_dir, srt_filename)
            
            # Save SRT file
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            
            print(f"✓ SRT file saved: {srt_path}")
        except Exception as e:
            print(f"⚠️  Could not save SRT file: {e}")
    
    @staticmethod
    def _faster_whisper_to_srt(segments) -> str:
        """
        Convert faster-whisper transcription segments to SRT format.
        
        Args:
            segments: Iterator of faster-whisper segments
            
        Returns:
            SRT formatted string
        """
        srt_lines = []
        
        for i, segment in enumerate(segments, start=1):
            start_time = segment.start
            end_time = segment.end
            text = segment.text.strip()
            
            # Format timestamps
            start_srt = KaraokeGenerator._seconds_to_srt_timestamp(start_time)
            end_srt = KaraokeGenerator._seconds_to_srt_timestamp(end_time)
            
            # Add SRT entry
            srt_lines.append(f"{i}")
            srt_lines.append(f"{start_srt} --> {end_srt}")
            srt_lines.append(text)
            srt_lines.append("")  # Blank line between entries
        
        return "\n".join(srt_lines)
    
    @staticmethod
    def normalize_srt_to_zero(srt_content: str) -> str:
        """
        Normalize SRT timestamps so the first subtitle starts at 00:00:00.
        
        Args:
            srt_content: SRT content string
            
        Returns:
            Normalized SRT content
        """
        import re
        
        # Find first timestamp
        timestamp_pattern = r'(\d{2}):(\d{2}):(\d{2}),(\d{3})'
        matches = list(re.finditer(timestamp_pattern, srt_content))
        
        if not matches:
            return srt_content
        
        # Get first start time
        first_match = matches[0]
        hours = int(first_match.group(1))
        minutes = int(first_match.group(2))
        seconds = int(first_match.group(3))
        milliseconds = int(first_match.group(4))
        
        # Calculate offset in seconds
        first_time_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
        
        if first_time_seconds == 0:
            return srt_content  # Already starts at 0
        
        # Shift all timestamps back by first_time_seconds
        from src.karaoke_generator import KaraokeGenerator
        return KaraokeGenerator.adjust_srt_timing(srt_content, -first_time_seconds)
    
    @staticmethod
    def adjust_srt_timing(srt_content: str, offset_seconds: float) -> str:
        """
        Adjust all timestamps in an SRT file by adding an offset.
        
        Args:
            srt_content: SRT file content as string
{{ ... }}
            offset_seconds: Seconds to add (can be negative)
            
        Returns:
            Modified SRT content with adjusted timestamps
        """
        import re
        
        def adjust_timestamp(match):
            start = match.group(1)
            end = match.group(2)
            
            # Parse timestamps
            def parse_srt_time(ts):
                h, m, s_ms = ts.split(':')
                s, ms = s_ms.split(',')
                return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
            
            def format_srt_time(seconds):
                if seconds < 0:
                    seconds = 0
                h = int(seconds // 3600)
                m = int((seconds % 3600) // 60)
                s = int(seconds % 60)
                ms = int((seconds % 1) * 1000)
                return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
            
            start_sec = parse_srt_time(start) + offset_seconds
            end_sec = parse_srt_time(end) + offset_seconds
            
            return f"{format_srt_time(start_sec)} --> {format_srt_time(end_sec)}"
        
        # Pattern to match SRT timestamps
        pattern = r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})'
        adjusted = re.sub(pattern, adjust_timestamp, srt_content)
        
        return adjusted
    
    @staticmethod
    def _seconds_to_srt_timestamp(seconds: float) -> str:
        """
        Convert seconds to SRT timestamp format.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            SRT timestamp string (HH:MM:SS,mmm)
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    @staticmethod
    def parse_srt_timestamp(ts_str: str) -> float:
        """
        Parse SRT timestamp to seconds.
        
        Args:
            ts_str: Timestamp string in SRT format (HH:MM:SS,mmm)
            
        Returns:
            Time in seconds
        """
        h, m, s_milli = ts_str.split(':')
        s, milli = s_milli.split(',')
        h, m, s, milli = int(h), int(m), int(s), int(milli)
        return h * 3600 + m * 60 + s + milli / 1000.0
    
    @staticmethod
    def format_timestamp_ass(seconds: float) -> str:
        """
        Format seconds to ASS timestamp.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Timestamp string in ASS format (H:MM:SS.cc)
        """
        cs = int((seconds % 1) * 100)
        s = int(seconds)
        h, s = divmod(s, 3600)
        m, s = divmod(s, 60)
        return f"{h}:{m:02}:{s:02}.{cs:02}"
    
    @staticmethod
    def detect_crop(
        infile: str,
        start_offset: int = 30,
        frame_count: int = 300
    ) -> Optional[str]:
        """
        Detect crop parameters for a video.
        
        Args:
            infile: Input video file path
            start_offset: Seconds to skip before analyzing
            frame_count: Number of frames to analyze
            
        Returns:
            Crop parameters string or None
        """
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_offset),
            "-i", infile,
            "-vf", "cropdetect=round=2:reset=0",
            "-frames:v", str(frame_count),
            "-f", "null", "-"
        ]
        result = subprocess.run(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        pattern = r'crop=\d+:\d+:\d+:\d+'
        matches = re.findall(pattern, result.stderr)
        if matches:
            return matches[-1]
        return None
    
    def srt_to_ass_karaoke(
        self,
        srt_content: str,
        ass_path: str,
        progressive_fill: bool = True,
        primary_colour: str = "&H0000FFFF",
        secondary_colour: str = "&H00CCCCCC",
        outline_colour: str = "&HA64DFF",
    ) -> bool:
        """
        Convert SRT content to ASS karaoke format.
        
        Args:
            srt_content: SRT subtitle content
            ass_path: Output ASS file path
            
        Returns:
            True if successful, False otherwise
        """
        print(f"Converting SRT to ASS karaoke format...")

        karaoke_tag = "\\kf" if progressive_fill else "\\k"
        
        lines = srt_content.strip().split('\n')
        segments = []
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            if line.isdigit():
                idx = int(line)
                i += 1
                if i >= len(lines):
                    break
                time_line = lines[i].strip()
                i += 1
                text_lines = []
                while i < len(lines) and lines[i].strip():
                    text_lines.append(lines[i].strip())
                    i += 1
                i += 1  # Skip blank line
                if ' --> ' not in time_line:
                    continue
                start_str, end_str = time_line.split(' --> ')
                start = self.parse_srt_timestamp(start_str)
                end = self.parse_srt_timestamp(end_str)
                text = ' '.join(text_lines)
                segments.append((idx, start, end, text))
            else:
                i += 1
        
        center_x = self.video_width // 2
        pos_y_top = int(self.video_height * 0.33)
        pos_y_bottom = int(self.video_height * 0.72)
        
        with open(ass_path, 'w', encoding='utf-8') as f:
            f.write("[Script Info]\n")
            f.write("Title: Karaoke with Space Mono\n")
            f.write("ScriptType: v4.00+\n")
            f.write("Collisions: Normal\n")
            f.write(f"PlayResX: {self.video_width}\n")
            f.write(f"PlayResY: {self.video_height}\n")
            f.write("Timer: 100.0000\n\n")
            
            f.write("[V4+ Styles]\n")
            f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
                    "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
                    "Alignment, MarginL, MarginR, MarginV, Encoding\n")
            # Bottom style (alignment 2 = bottom center) - extremely close to middle
            # Outline color: #ff3a52 -> BGR format: &H523AFF
            f.write(f"Style: Main,{self.font_path},76,{primary_colour},{secondary_colour},{outline_colour},&H64000000,"
                    "1,0,0,0,100,100,0,0,1,3,0,2,480,480,450,1\n")
            f.write(f"Style: Shadow,{self.font_path},76,&H00FFFFFF,&HFFFFFFFF,&HFFFFFF,&H64000000,"
                    "1,0,0,0,100,100,0,0,1,0,0,2,480,480,450,1\n")
            # Top style (alignment 8 = top center) - extremely close to middle
            f.write(f"Style: MainTop,{self.font_path},76,{primary_colour},{secondary_colour},{outline_colour},&H64000000,"
                    "1,0,0,0,100,100,0,0,1,3,0,8,480,480,450,1\n")
            f.write(f"Style: ShadowTop,{self.font_path},76,&H00FFFFFF,&HFFFFFFFF,&HFFFFFF,&H64000000,"
                    "1,0,0,0,100,100,0,0,1,0,0,8,480,480,450,1\n\n")
            
            f.write("[Events]\n")
            f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
            
            for idx, start, end, text in segments:
                if not text.strip():
                    continue
                start_ass = self.format_timestamp_ass(start)
                end_ass = self.format_timestamp_ass(end)
                line_duration = end - start
                
                words = text.split()
                if not words:
                    continue
                
                total_cs = int(line_duration * 100)
                per_word_cs = max(total_cs // len(words), 1)
                remainder = total_cs - (per_word_cs * len(words))
                word_ks = [per_word_cs] * len(words)
                if remainder > 0:
                    word_ks[-1] += remainder
                
                # Alternate between bottom (even idx) and top (odd idx)
                is_top = (idx % 2 == 1)
                style_main = "MainTop" if is_top else "Main"
                style_shadow = "ShadowTop" if is_top else "Shadow"
                pos_y = pos_y_top if is_top else pos_y_bottom
                pos_tag = f"\\an5\\pos({center_x},{pos_y})"
                
                # Calculate duration in milliseconds for smooth growth
                duration_ms = int(line_duration * 1000)
                
                # Scale animation: smoothly grow from 100% to 120% over the entire duration
                # \org sets the origin point for scaling (center of text)
                main_line = "{\\fad(500,500)" + pos_tag + "\\fscx100\\fscy100\\t(0," + str(duration_ms) + ",\\fscx120\\fscy120)}"
                for w, kcs in zip(words, word_ks):
                    main_line += "{" + karaoke_tag + str(kcs) + "}" + str(w) + " "
                main_line = main_line.strip()
                
                # Glow effect - blurred shadow layer with same animation
                shadow_line = "{\\fad(500,500)" + pos_tag + "\\blur10\\3c&H000000&\\4c&H000000&\\fscx100\\fscy100\\t(0," + str(duration_ms) + ",\\fscx120\\fscy120)}"
                for w, kcs in zip(words, word_ks):
                    shadow_line += "{" + karaoke_tag + str(kcs) + "}" + str(w) + " "
                shadow_line = shadow_line.strip()
                
                f.write(f"Dialogue: 0,{start_ass},{end_ass},{style_shadow},,0,0,0,,{shadow_line}\n")
                f.write(f"Dialogue: 1,{start_ass},{end_ass},{style_main},,0,0,0,,{main_line}\n")
        
        print(f"ASS file created at '{ass_path}'.")
        return True
    
    def overlay_subtitles_and_logo(
        self,
        video_path: str,
        ass_path: str,
        output_video_path: str,
        logo_path: Optional[str] = None,
        fast_mode: bool = False,
        effect_overlay_path: Optional[str] = None,
        max_duration_seconds: Optional[float] = None,
        overlay_hue_deg: Optional[float] = None,
        overlay_saturation: Optional[float] = None,
        tint_color_hex: Optional[str] = None
    ) -> bool:
        """
        Overlay subtitles and logo on video.
        
        Args:
            video_path: Input video file path
            ass_path: ASS subtitle file path
            output_video_path: Output video file path
            logo_path: Optional logo file path (uses default if not provided)
            fast_mode: If True, use faster encoding settings (lower quality)
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(video_path):
            print(f"Video file '{video_path}' does not exist.")
            return False
        
        logo_path = logo_path or self.logo_path
        if not os.path.exists(logo_path):
            print(f"Logo file '{logo_path}' does not exist.")
            return False
        
        if not os.path.exists(ass_path):
            print(f"ASS file '{ass_path}' does not exist.")
            return False
        
        has_effect = False
        overlay_prepped = False  # Whether overlay already matches size/fps (e.g., pre-transcoded)
        if effect_overlay_path and os.path.exists(effect_overlay_path):
            has_effect = True
            # Probe overlay to see if it matches target resolution and ~30fps
            try:
                probe = subprocess.run(
                    [
                        "ffprobe", "-v", "error",
                        "-select_streams", "v:0",
                        "-show_entries", "stream=width,height,avg_frame_rate",
                        "-of", "json",
                        effect_overlay_path,
                    ],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", check=True
                )
                info = json.loads(probe.stdout)
                streams = info.get("streams", [])
                if streams:
                    st = streams[0]
                    w = int(st.get("width", 0) or 0)
                    h = int(st.get("height", 0) or 0)
                    fr = st.get("avg_frame_rate", "0/1")
                    if isinstance(fr, str) and "/" in fr:
                        num, den = fr.split("/")
                        try:
                            num = int(num)
                            den = int(den)
                            fps = (num / den) if den else 0.0
                        except Exception:
                            fps = 0.0
                    else:
                        fps = 0.0
                    # Consider it prepped if dimensions match and fps is ~24 or ~30
                    if (
                        w == self.video_width and h == self.video_height and
                        (abs(fps - 30.0) <= 1.0 or abs(fps - 24.0) <= 1.0)
                    ):
                        overlay_prepped = True
                        print("✓ Overlay is pre-transcoded to target size/fps; skipping scale/fps filters")
            except Exception:
                pass
        
        crop_params = self.detect_crop(video_path, start_offset=30, frame_count=300)
        if not crop_params:
            print("Could not detect crop parameters. Proceeding without cropping.")
            crop_params = None  # Don't apply crop if detection failed
        else:
            print(f"Detected crop parameters: {crop_params}")
        
        # Escape special characters in ASS path for FFmpeg filter
        # FFmpeg requires : \ ' , and [ ] to be escaped in filter paths
        ass_path_escaped = (ass_path
            .replace('\\', '\\\\')
            .replace(':', '\\:')
            .replace("'", "\\'")
            .replace(',', '\\,')
            .replace('[', '\\[')
            .replace(']', '\\]')
        )

        # Use the lighter-weight ASS filter when we already have an .ass file.
        # The subtitles filter may require additional conversion via libav*.
        subtitles_filter_name = "ass" if str(ass_path).lower().endswith(".ass") else "subtitles"
        
        # Build filter complex based on whether crop is available and if we have an effect layer
        drawbox_color = tint_color_hex if tint_color_hex else "black"
        logo_stream = 2 if has_effect else 1
        logo_label = f"[{logo_stream}:v]"
        if has_effect:
            if crop_params:
                # With crop + darkening + effect blended before subtitles
                if overlay_prepped:
                    filter_complex = (
                        f"[0:v]{crop_params},"
                        f"scale={self.video_width}:{self.video_height}:force_original_aspect_ratio=increase:flags=fast_bilinear,"
                        f"crop={self.video_width}:{self.video_height},"
                        f"eq=brightness=-0.22:contrast=0.9,drawbox=x=0:y=0:w=iw:h=ih:color={drawbox_color}@0.15:t=fill[dimmed];"
                        f"[1:v]"
                        + (((f"hue=h={overlay_hue_deg}*PI/180" + (f":s={overlay_saturation}" if overlay_saturation is not None else "") + ",") ) if overlay_hue_deg is not None else "")
                        + "setsar=1,setpts=PTS-STARTPTS,format=rgba,colorchannelmixer=aa=0.45[fx];"
                        f"[dimmed][fx]overlay=shortest=1[bgfx];"
                        f"[bgfx]setsar=1,setpts=PTS-STARTPTS[subbed];"
                        f"[subbed]{subtitles_filter_name}={ass_path_escaped}[subbed_with_subs];"
                        f"{logo_label}scale={self.logo_width}:{self.logo_height}:flags=fast_bilinear,format=rgba[logoScaled];[subbed_with_subs][logoScaled]overlay=x=W-w-25:y=25[finalv]"
                    )
                else:
                    filter_complex = (
                        f"[0:v]{crop_params},"
                        f"scale={self.video_width}:{self.video_height}:force_original_aspect_ratio=increase:flags=fast_bilinear,"
                        f"crop={self.video_width}:{self.video_height},"
                        f"eq=brightness=-0.22:contrast=0.9,drawbox=x=0:y=0:w=iw:h=ih:color={drawbox_color}@0.15:t=fill[dimmed];"
                        f"[1:v]"
                        + (((f"hue=h={overlay_hue_deg}*PI/180" + (f":s={overlay_saturation}" if overlay_saturation is not None else "") + ",") ) if overlay_hue_deg is not None else "")
                        + f"scale={self.video_width}:{self.video_height}:force_original_aspect_ratio=increase:flags=fast_bilinear,"
                        f"crop={self.video_width}:{self.video_height},fps=30,setsar=1,setpts=PTS-STARTPTS,format=rgba,colorchannelmixer=aa=0.45[fx];"
                        f"[dimmed][fx]overlay=shortest=1[bgfx];"
                        f"[bgfx]setsar=1,setpts=PTS-STARTPTS[subbed];"
                        f"[subbed]{subtitles_filter_name}={ass_path_escaped}[subbed_with_subs];"
                        f"{logo_label}scale={self.logo_width}:{self.logo_height}:flags=fast_bilinear,format=rgba[logoScaled];[subbed_with_subs][logoScaled]overlay=x=W-w-25:y=25[finalv]"
                    )
            else:
                # Without crop - pad + darken + effect blend before subtitles
                if overlay_prepped:
                    filter_complex = (
                        f"[0:v]scale={self.video_width}:{self.video_height}:force_original_aspect_ratio=decrease:flags=fast_bilinear,"
                        f"pad={self.video_width}:{self.video_height}:(ow-iw)/2:(oh-ih)/2,"
                        f"eq=brightness=-0.22:contrast=0.9,drawbox=x=0:y=0:w=iw:h=ih:color={drawbox_color}@0.15:t=fill[dimmed];"
                        f"[1:v]"
                        + (((f"hue=h={overlay_hue_deg}*PI/180" + (f":s={overlay_saturation}" if overlay_saturation is not None else "") + ",") ) if overlay_hue_deg is not None else "")
                        + "setsar=1,setpts=PTS-STARTPTS,format=rgba,colorchannelmixer=aa=0.45[fx];"
                        f"[dimmed][fx]overlay=shortest=1[bgfx];"
                        f"[bgfx]setsar=1,setpts=PTS-STARTPTS[subbed];"
                        f"[subbed]{subtitles_filter_name}={ass_path_escaped}[subbed_with_subs];"
                        f"{logo_label}scale={self.logo_width}:{self.logo_height}:flags=fast_bilinear,format=rgba[logoScaled];[subbed_with_subs][logoScaled]overlay=x=W-w-25:y=25[finalv]"
                    )
                else:
                    filter_complex = (
                        f"[0:v]scale={self.video_width}:{self.video_height}:force_original_aspect_ratio=decrease:flags=fast_bilinear,"
                        f"pad={self.video_width}:{self.video_height}:(ow-iw)/2:(oh-ih)/2,"
                        f"eq=brightness=-0.22:contrast=0.9,drawbox=x=0:y=0:w=iw:h=ih:color={drawbox_color}@0.15:t=fill[dimmed];"
                        f"[1:v]"
                        + (((f"hue=h={overlay_hue_deg}*PI/180" + (f":s={overlay_saturation}" if overlay_saturation is not None else "") + ",") ) if overlay_hue_deg is not None else "")
                        + f"scale={self.video_width}:{self.video_height}:force_original_aspect_ratio=increase:flags=fast_bilinear,"
                        f"crop={self.video_width}:{self.video_height},fps=30,setsar=1,setpts=PTS-STARTPTS,format=rgba,colorchannelmixer=aa=0.45[fx];"
                        f"[dimmed][fx]overlay=shortest=1[bgfx];"
                        f"[bgfx]setsar=1,setpts=PTS-STARTPTS[subbed];"
                        f"[subbed]{subtitles_filter_name}={ass_path_escaped}[subbed_with_subs];"
                        f"{logo_label}scale={self.logo_width}:{self.logo_height}:flags=fast_bilinear,format=rgba[logoScaled];[subbed_with_subs][logoScaled]overlay=x=W-w-25:y=25[finalv]"
                    )
        else:
            if crop_params:
                # With crop + dark overlay
                filter_complex = (
                    f"[0:v]{crop_params},"
                    f"scale={self.video_width}:{self.video_height}:force_original_aspect_ratio=increase:flags=fast_bilinear,"
                    f"crop={self.video_width}:{self.video_height},"
                    f"eq=brightness=-0.22:contrast=0.9,drawbox=x=0:y=0:w=iw:h=ih:color={drawbox_color}@0.15:t=fill[dimmed];"
                    f"[dimmed]setsar=1,setpts=PTS-STARTPTS[subbed];"
                    f"[subbed]{subtitles_filter_name}={ass_path_escaped}[subbed_with_subs];"
                    f"{logo_label}scale={self.logo_width}:{self.logo_height}:flags=fast_bilinear,format=rgba[logoScaled];[subbed_with_subs][logoScaled]overlay=x=W-w-25:y=25[finalv]"
                )
            else:
                # Without crop - just scale and add subtitles + dark overlay
                filter_complex = (
                    f"[0:v]scale={self.video_width}:{self.video_height}:force_original_aspect_ratio=decrease:flags=fast_bilinear,"
                    f"pad={self.video_width}:{self.video_height}:(ow-iw)/2:(oh-ih)/2,"
                    f"eq=brightness=-0.22:contrast=0.9,drawbox=x=0:y=0:w=iw:h=ih:color={drawbox_color}@0.15:t=fill[dimmed];"
                    f"[dimmed]setsar=1,setpts=PTS-STARTPTS[subbed];"
                    f"[subbed]{subtitles_filter_name}={ass_path_escaped}[subbed_with_subs];"
                    f"{logo_label}scale={self.logo_width}:{self.logo_height}:flags=fast_bilinear,format=rgba[logoScaled];[subbed_with_subs][logoScaled]overlay=x=W-w-25:y=25[finalv]"
                )
        
        # Choose encoding settings based on mode
        if fast_mode:
            # Fast mode: prefer hardware encode if available, else ultrafast x264
            preset = "ultrafast"
            crf = "33"  # More aggressive for speed in software fallback
            audio_bitrate = "192k"
            video_codec = "h264_videotoolbox"
            use_hwenc = True
            print("⚡ Using FAST mode (hardware encode if available, else ultrafast)")
        else:
            # Normal mode: balanced quality/speed
            preset = "medium"
            crf = "23"
            audio_bitrate = "320k"
            video_codec = "libx264"
            use_hwenc = False
        
        # Build codec args (hardware for fast mode when available)
        if use_hwenc:
            # Hardware encode (macOS): h264_videotoolbox, fallback to libx264 if not supported at runtime
            video_codec_args = ["-c:v", video_codec, "-b:v", "4000k"]
            # Force 24fps output for drafts to reduce frames encoded
            fps_args = ["-r", "24"]
        else:
            video_codec_args = ["-c:v", "libx264", "-crf", crf, "-preset", preset]
            fps_args = []

        # Duration handling: if max_duration_seconds is given, trim to it; else rely on -shortest
        duration_args = ["-t", f"{max_duration_seconds:.6f}"] if max_duration_seconds and max_duration_seconds > 0 else ["-shortest"]

        common_tail = [
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", audio_bitrate,
            "-movflags", "+faststart",
            output_video_path
        ]

        loop_bg = bool(max_duration_seconds and max_duration_seconds > 0)
        video_input_args = ["-stream_loop", "-1", "-i", video_path] if loop_bg else ["-i", video_path]

        if has_effect:
            command = [
                "ffmpeg", "-y",
                *video_input_args,
                "-stream_loop", "-1", "-i", effect_overlay_path,
                "-i", logo_path,
                "-filter_complex", filter_complex,
                "-map", "[finalv]",
                "-map", "0:a?",
            ] + fps_args + video_codec_args + duration_args + common_tail
        else:
            command = [
                "ffmpeg", "-y",
                *video_input_args,
                "-i", logo_path,
                "-filter_complex", filter_complex,
                "-map", "[finalv]",
                "-map", "0:a?",
            ] + fps_args + video_codec_args + duration_args + common_tail
        
        print("Running FFmpeg command...")
        try:
            subprocess.run(command, check=True)
            print(f"Karaoke video created at '{output_video_path}'.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error running FFmpeg: {e}")
            return False
    
    def generate_karaoke(self, video_path: str, srt_content: Optional[str] = None, output_path: str = None, logo_path: Optional[str] = None, fast_mode: bool = False, ass_file: Optional[str] = None, effect_overlay_paths: Optional[List[str]] = None, max_duration_seconds: Optional[float] = None, overlay_hue_deg: Optional[float] = None, overlay_saturation: Optional[float] = None, overlay_tint_color: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Generate a karaoke video from a video file and SRT content or ASS file.
        
        Args:
            video_path: Input video file path
            srt_content: SRT subtitle content (optional if ass_file provided)
            output_path: Output video file path
            logo_path: Optional logo file path
            fast_mode: If True, use faster encoding settings (lower quality)
            ass_file: Optional pre-generated ASS file (for word-level karaoke)
            
        Returns:
            Tuple of (success, output_path or error_message)
        """
        delete_ass_on_exit = False
        ass_path: Optional[str] = None
        try:
            # Use provided ASS file or convert SRT to ASS
            if ass_file and os.path.exists(ass_file):
                # Use the provided ASS file directly
                ass_path = ass_file
            elif srt_content:
                # Create temporary ASS file with sanitized name (no special chars)
                import re
                # Sanitize output path for ASS file (remove apostrophes, commas, and special chars)
                ass_path = output_path.replace('.mp4', '.ass')
                ass_path = ass_path.replace("'", "").replace('"', '').replace(',', '')
                delete_ass_on_exit = True
                
                # Convert SRT to ASS
                if not self.srt_to_ass_karaoke(srt_content, ass_path):
                    return False, "Failed to convert SRT to ASS"
            else:
                return False, "Either srt_content or ass_file must be provided"
            
            # Randomly choose an effect overlay if available
            effect_path = None
            candidates = []
            if effect_overlay_paths:
                candidates = [p for p in effect_overlay_paths if p and os.path.exists(p)]
            else:
                # Enforce prepped overlays only: use /overlays directory exclusively
                overlay_dir = "/Users/lekin/Code/tout-baigne/overlays"
                if os.path.isdir(overlay_dir):
                    exts = {".mp4", ".mov", ".m4v", ".webm"}
                    for name in os.listdir(overlay_dir):
                        p = os.path.join(overlay_dir, name)
                        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in exts:
                            candidates.append(p)
            if candidates:
                pool = [p for p in candidates if p != self._last_effect_path] or candidates
                effect_path = random.choice(pool)
                self._last_effect_path = effect_path
                print(f"✨ Using overlay effect: {os.path.basename(effect_path)}")

            # Overlay subtitles, optional effect, and logo
            success = self.overlay_subtitles_and_logo(
                video_path, ass_path, output_path, logo_path, fast_mode=fast_mode, effect_overlay_path=effect_path, max_duration_seconds=max_duration_seconds, overlay_hue_deg=overlay_hue_deg, overlay_saturation=overlay_saturation, tint_color_hex=overlay_tint_color
            )
            if not success:
                return False, "Failed to overlay subtitles and logo"
            
            return True, output_path
            
        except Exception as e:
            return False, str(e)
        finally:
            if delete_ass_on_exit and ass_path and os.path.exists(ass_path):
                try:
                    os.unlink(ass_path)
                except Exception:
                    pass
    def align_audio_by_chroma_dtw(self, video_path: str, audio_path: str) -> dict:
        """
        Align audio using beat-synchronous chroma features + DTW.
        This is the most robust method for music alignment.
        
        Method:
        1. Extract chroma features (pitch content) from both audios
        2. Synchronize to beat positions for robustness
        3. Use DTW to find optimal alignment path
        4. Determine constant offset OR global tempo stretch
        
        Args:
            video_path: Path to video file (YouTube clip)
            audio_path: Path to audio file (original track)
            
        Returns:
            Dict with alignment info: {
                'offset': float,  # Constant time offset (seconds)
                'tempo_ratio': float,  # Global tempo stretch factor
                'confidence': float,  # Match quality (0-1)
                'trim_start': float,  # Where to start trimming video
                'alignment_type': str  # 'offset' or 'stretch'
            }
        """
        try:
            import subprocess
            import numpy as np
            from scipy import signal
            from scipy.spatial.distance import cdist
            
            print("🎵 Aligning audio using beat-synchronous chroma + DTW...")
            
            # Extract audio arrays
            def extract_audio_array(file_path):
                cmd = [
                    "ffmpeg", "-i", file_path, "-vn",
                    "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1",
                    "-f", "wav", "pipe:1"
                ]
                result = subprocess.run(cmd, capture_output=True, check=True)
                audio_data = np.frombuffer(result.stdout[44:], dtype=np.int16)
                return audio_data.astype(np.float32) / 32768.0
            
            # Compute chroma features (12-dimensional pitch class profiles)
            def compute_chroma(audio, sr=22050, hop_length=512):
                """Compute chromagram using FFT"""
                n_fft = 4096
                n_chroma = 12
                
                # Frequency bins to chroma mapping
                freqs = np.fft.rfftfreq(n_fft, 1/sr)
                
                # A4 = 440 Hz, 12 semitones per octave
                chroma_filters = np.zeros((n_chroma, len(freqs)))
                for i, freq in enumerate(freqs):
                    if freq > 0:
                        # Convert frequency to MIDI note
                        midi = 12 * np.log2(freq / 440.0) + 69
                        chroma_bin = int(np.round(midi)) % 12
                        chroma_filters[chroma_bin, i] = 1.0
                
                # Compute STFT and map to chroma
                chroma = []
                for i in range(0, len(audio) - n_fft, hop_length):
                    frame = audio[i:i+n_fft] * np.hanning(n_fft)
                    spectrum = np.abs(np.fft.rfft(frame))
                    chroma_frame = chroma_filters @ spectrum
                    # Normalize
                    chroma_frame = chroma_frame / (np.sum(chroma_frame) + 1e-10)
                    chroma.append(chroma_frame)
                
                return np.array(chroma).T  # Shape: (12, n_frames)
            
            # Simple beat detection using onset strength
            def detect_beats(audio, sr=22050, hop_length=512):
                """Detect beat positions using onset strength envelope"""
                # Compute onset strength (spectral flux)
                n_fft = 2048
                onset_env = []
                prev_spectrum = None
                
                for i in range(0, len(audio) - n_fft, hop_length):
                    frame = audio[i:i+n_fft] * np.hanning(n_fft)
                    spectrum = np.abs(np.fft.rfft(frame))
                    
                    if prev_spectrum is not None:
                        # Spectral flux (positive differences)
                        flux = np.sum(np.maximum(0, spectrum - prev_spectrum))
                        onset_env.append(flux)
                    else:
                        onset_env.append(0)
                    
                    prev_spectrum = spectrum
                
                onset_env = np.array(onset_env)
                
                # Find peaks (beats)
                # Simple peak detection with minimum distance
                min_beat_distance = int(0.3 * sr / hop_length)  # ~300ms between beats
                
                beats = []
                for i in range(min_beat_distance, len(onset_env) - min_beat_distance):
                    if onset_env[i] > np.mean(onset_env) * 1.5:  # Threshold
                        # Check if it's a local maximum
                        if onset_env[i] == np.max(onset_env[i-min_beat_distance:i+min_beat_distance]):
                            beats.append(i)
                
                return np.array(beats) if beats else np.arange(0, len(onset_env), min_beat_distance)
            
            # DTW alignment
            def dtw_alignment(chroma1, chroma2):
                """Dynamic Time Warping to find optimal alignment path"""
                # Compute cost matrix (cosine distance)
                cost_matrix = cdist(chroma1.T, chroma2.T, metric='cosine')
                
                # DTW dynamic programming
                n, m = cost_matrix.shape
                dtw_matrix = np.full((n+1, m+1), np.inf)
                dtw_matrix[0, 0] = 0
                
                for i in range(1, n+1):
                    for j in range(1, m+1):
                        cost = cost_matrix[i-1, j-1]
                        dtw_matrix[i, j] = cost + min(
                            dtw_matrix[i-1, j],    # insertion
                            dtw_matrix[i, j-1],    # deletion
                            dtw_matrix[i-1, j-1]   # match
                        )
                
                # Backtrack to find path
                path = []
                i, j = n, m
                while i > 0 and j > 0:
                    path.append((i-1, j-1))
                    
                    # Choose minimum predecessor
                    candidates = [
                        (i-1, j-1, dtw_matrix[i-1, j-1]),
                        (i-1, j, dtw_matrix[i-1, j]),
                        (i, j-1, dtw_matrix[i, j-1])
                    ]
                    i, j, _ = min(candidates, key=lambda x: x[2])
                
                path.reverse()
                return np.array(path), dtw_matrix[n, m] / (n + m)
            
            print("   Extracting audio...")
            video_audio = extract_audio_array(video_path)
            original_audio = extract_audio_array(audio_path)
            
            sr = 22050
            hop_length = 512
            
            video_duration = len(video_audio) / sr
            audio_duration = len(original_audio) / sr
            
            print(f"   Video: {video_duration:.1f}s, Audio: {audio_duration:.1f}s")
            
            # Compute chroma features
            print("   Computing chroma features...")
            video_chroma = compute_chroma(video_audio, sr, hop_length)
            audio_chroma = compute_chroma(original_audio, sr, hop_length)
            
            print(f"   Video chroma: {video_chroma.shape}")
            print(f"   Audio chroma: {audio_chroma.shape}")
            
            # Detect beats
            print("   Detecting beats...")
            video_beats = detect_beats(video_audio, sr, hop_length)
            audio_beats = detect_beats(original_audio, sr, hop_length)
            
            print(f"   Video beats: {len(video_beats)}, Audio beats: {len(audio_beats)}")
            
            # Beat-synchronous chroma (average chroma between beats)
            def beat_sync_chroma(chroma, beats):
                sync_chroma = []
                for i in range(len(beats) - 1):
                    start, end = beats[i], beats[i+1]
                    sync_chroma.append(np.mean(chroma[:, start:end], axis=1))
                return np.array(sync_chroma).T
            
            video_chroma_sync = beat_sync_chroma(video_chroma, video_beats)
            audio_chroma_sync = beat_sync_chroma(audio_chroma, audio_beats)
            
            print(f"   Beat-sync chroma: Video {video_chroma_sync.shape}, Audio {audio_chroma_sync.shape}")
            
            # Use first 60 seconds for alignment (or full if shorter)
            max_frames = min(120, video_chroma_sync.shape[1], audio_chroma_sync.shape[1])
            
            print("   Running DTW alignment...")
            path, dtw_cost = dtw_alignment(
                audio_chroma_sync[:, :max_frames],
                video_chroma_sync[:, :max_frames]
            )
            
            print(f"   DTW cost: {dtw_cost:.4f}")
            
            # Analyze alignment path to determine offset vs tempo stretch
            path_video = path[:, 1]
            path_audio = path[:, 0]
            
            # Linear regression to find tempo ratio
            if len(path) > 10:
                # Fit line: video_frame = slope * audio_frame + intercept
                coeffs = np.polyfit(path_audio, path_video, 1)
                tempo_ratio = coeffs[0]
                offset_frames = coeffs[1]
            else:
                tempo_ratio = 1.0
                offset_frames = path[0, 1] - path[0, 0]
            
            # Convert to time
            frame_duration = hop_length / sr
            offset_seconds = offset_frames * frame_duration
            
            # Map beat frames to time
            if len(video_beats) > 0:
                offset_seconds = (video_beats[int(offset_frames)] * hop_length) / sr if offset_frames < len(video_beats) else offset_seconds
            
            confidence = 1.0 / (1.0 + dtw_cost)  # Convert cost to confidence
            
            # Decide: offset only or tempo stretch?
            if abs(tempo_ratio - 1.0) < 0.02:
                # < 2% tempo difference: use constant offset only
                alignment_type = 'offset'
                tempo_ratio = 1.0
                print(f"   ✅ Constant offset alignment: {offset_seconds:.2f}s")
            else:
                # Significant tempo difference: use stretch
                alignment_type = 'stretch'
                print(f"   ⚙️  Tempo stretch alignment: {tempo_ratio:.4f}x, offset: {offset_seconds:.2f}s")
            
            print(f"   Confidence: {confidence:.3f}")
            
            return {
                'offset': max(0, offset_seconds),
                'tempo_ratio': tempo_ratio,
                'confidence': confidence,
                'trim_start': max(0, offset_seconds),
                'alignment_type': alignment_type
            }
            
        except Exception as e:
            print(f"⚠️  Chroma+DTW alignment failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'offset': 0.0,
                'tempo_ratio': 1.0,
                'confidence': 0.0,
                'trim_start': 0.0,
                'alignment_type': 'fallback'
            }
    
    def align_audio_by_fingerprinting(self, video_path: str, audio_path: str) -> dict:
        """
        Align video and audio by comparing their full waveforms.
        Uses spectral fingerprinting and cross-correlation across the entire duration.
        
        Args:
            video_path: Path to video file (YouTube clip)
            audio_path: Path to audio file (original track)
            
        Returns:
            Dict with alignment info: {
                'offset': float,  # Where audio starts in video (seconds)
                'speed_ratio': float,  # Video speed / audio speed
                'confidence': float,  # Match quality (0-1)
                'trim_start': float,  # Where to start trimming video
                'trim_end': float  # Where to end trimming video
            }
        """
        try:
            import subprocess
            import numpy as np
            from scipy import signal
            
            print("🔍 Analyzing full audio waveforms for alignment...")
            
            # Extract audio from both sources
            def extract_audio_array(file_path):
                cmd = [
                    "ffmpeg", "-i", file_path, "-vn",
                    "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                    "-f", "wav", "pipe:1"
                ]
                result = subprocess.run(cmd, capture_output=True, check=True)
                audio_data = np.frombuffer(result.stdout[44:], dtype=np.int16)
                return audio_data.astype(np.float32) / 32768.0
            
            # Extract spectral features (more robust than raw waveform)
            def compute_spectral_fingerprint(audio, sr=16000, hop_length=512):
                """Compute spectral centroid as fingerprint (robust to volume changes)"""
                # Simple spectrogram using FFT
                n_fft = 2048
                hop = hop_length
                
                # Compute short-time energy in frequency bands
                fingerprint = []
                for i in range(0, len(audio) - n_fft, hop):
                    frame = audio[i:i+n_fft]
                    # Apply window
                    frame = frame * np.hanning(n_fft)
                    # FFT
                    spectrum = np.abs(np.fft.rfft(frame))
                    # Spectral centroid (weighted average of frequencies)
                    freqs = np.arange(len(spectrum))
                    centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-10)
                    fingerprint.append(centroid)
                
                return np.array(fingerprint)
            
            video_audio = extract_audio_array(video_path)
            original_audio = extract_audio_array(audio_path)
            
            video_duration = len(video_audio) / 16000.0
            audio_duration = len(original_audio) / 16000.0
            
            print(f"   Video audio: {video_duration:.1f}s")
            print(f"   Original audio: {audio_duration:.1f}s")
            
            # Compute fingerprints
            print("   Computing spectral fingerprints...")
            video_fp = compute_spectral_fingerprint(video_audio)
            audio_fp = compute_spectral_fingerprint(original_audio)
            
            print(f"   Video fingerprint: {len(video_fp)} frames")
            print(f"   Audio fingerprint: {len(audio_fp)} frames")
            
            # Find offset using cross-correlation on fingerprints
            print("   Finding alignment...")
            
            # Use first 60 seconds of audio as reference
            ref_length = min(60 * 16000 // 512, len(audio_fp))
            reference = audio_fp[:ref_length]
            
            # Search in first 3 minutes of video
            search_length = min(180 * 16000 // 512, len(video_fp))
            search = video_fp[:search_length]
            
            # Cross-correlation
            correlation = signal.correlate(search, reference, mode='valid')
            best_offset_frames = np.argmax(correlation)
            best_offset_seconds = (best_offset_frames * 512) / 16000.0
            
            # Calculate confidence
            max_corr = correlation[best_offset_frames]
            mean_corr = np.mean(correlation)
            std_corr = np.std(correlation)
            confidence = float((max_corr - mean_corr) / (std_corr + 1e-10))
            confidence = min(1.0, confidence / 10.0)  # Normalize to 0-1
            
            # Estimate speed ratio by comparing durations
            # If video is longer, it might be slower or have intro/outro
            speed_ratio = 1.0
            if video_duration > audio_duration * 1.1:
                # Video is significantly longer - likely has intro/outro
                speed_ratio = 1.0
            else:
                # Similar durations - might have slight speed difference
                speed_ratio = video_duration / audio_duration
            
            # Calculate trim points
            trim_start = best_offset_seconds
            trim_end = trim_start + audio_duration
            
            print(f"✓ Alignment found:")
            print(f"   Offset: {best_offset_seconds:.2f}s")
            print(f"   Speed ratio: {speed_ratio:.4f}")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Trim: {trim_start:.2f}s → {trim_end:.2f}s")
            
            return {
                'offset': best_offset_seconds,
                'speed_ratio': speed_ratio,
                'confidence': confidence,
                'trim_start': trim_start,
                'trim_end': trim_end
            }
            
        except Exception as e:
            print(f"⚠️  Audio alignment failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'offset': 0.0,
                'speed_ratio': 1.0,
                'confidence': 0.0,
                'trim_start': 0.0,
                'trim_end': 0.0
            }
    
    def sync_video_to_audio(self, video_path: str, audio_path: str, output_path: str, manual_offset: float = 0.0, fast_mode: bool = False) -> bool:
        """
        Intelligent audio-based synchronization using waveform correlation.
        
        Strategy:
        1. Find exact audio offset using cross-correlation
        2. Trim video to match audio duration
        3. Replace video audio with original audio
        
        Args:
            video_path: Path to video file (YouTube clip)
            audio_path: Path to audio file (original track)
            output_path: Path for synchronized video output
            fast_mode: Use faster encoding settings
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import subprocess
            import tempfile
            
            # Get durations
            audio_probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
                capture_output=True, text=True, encoding="utf-8", errors="replace", check=True
            )
            audio_duration = float(audio_probe.stdout.strip())
            
            video_probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", video_path],
                capture_output=True, text=True, encoding="utf-8", errors="replace", check=True
            )
            video_duration = float(video_probe.stdout.strip())
            
            print(f"📊 Audio duration: {audio_duration:.1f}s")
            print(f"📊 Video duration: {video_duration:.1f}s")
            
            # Use manual offset if provided, otherwise use Chroma+DTW
            if manual_offset != 0:
                print(f"📊 Using manual offset: {manual_offset:+.1f}s")
                start_time = manual_offset
                tempo_ratio = 1.0
                alignment_type = 'manual'
            else:
                # Analyze audio using beat-synchronous chroma + DTW
                alignment = self.align_audio_by_chroma_dtw(video_path, audio_path)
                
                start_time = alignment['trim_start']
                tempo_ratio = alignment['tempo_ratio']
                confidence = alignment['confidence']
                alignment_type = alignment['alignment_type']
                
                if confidence < 0.1:
                    print(f"⚠️  Low confidence alignment, using fallback")
                    # Fallback: assume intro is the difference
                    duration_diff = video_duration - audio_duration
                    start_time = max(0, duration_diff / 2)
                    tempo_ratio = 1.0
                    alignment_type = 'fallback'
                
                if alignment_type == 'offset':
                    print(f"📊 Sync strategy: Constant offset ({start_time:.2f}s)")
                elif alignment_type == 'stretch':
                    print(f"📊 Sync strategy: Tempo stretch ({tempo_ratio:.4f}x) + offset ({start_time:.2f}s)")
                else:
                    print(f"📊 Sync strategy: Fallback trim from {start_time:.2f}s")
            
            # Build FFmpeg command based on detected alignment
            preset = "ultrafast" if fast_mode else "fast"
            crf = "28" if fast_mode else "23"
            audio_bitrate = "192k" if fast_mode else "320k"
            
            if abs(tempo_ratio - 1.0) > 0.02:
                # Need tempo stretch (time-stretch video to match audio tempo)
                print(f"⚙️  Applying tempo stretch: {tempo_ratio:.4f}x")
                # Inverse ratio for setpts (slower video = larger PTS multiplier)
                video_filter = f"setpts={1.0/tempo_ratio}*PTS"
                
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(start_time),  # Start at detected offset
                    "-i", video_path,
                    "-i", audio_path,
                    "-filter_complex", f"[0:v]{video_filter}[v]",
                    "-map", "[v]", "-map", "1:a:0",
                    "-t", str(audio_duration),  # Match audio duration
                    "-c:v", "libx264", "-preset", preset, "-crf", crf,
                    "-c:a", "aac", "-b:a", audio_bitrate,
                    output_path
                ]
            else:
                # Just trim and replace audio
                print(f"✂️  Trimming video: {start_time:.2f}s → {start_time + audio_duration:.2f}s")
                
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(start_time),  # Start at detected offset
                    "-i", video_path,
                    "-i", audio_path,
                    "-t", str(audio_duration),  # Match audio duration
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "libx264", "-preset", preset, "-crf", crf,
                    "-c:a", "aac", "-b:a", audio_bitrate,
                    output_path
                ]
            
            print("🔄 Synchronizing video with audio...")
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✅ Video synchronized: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error syncing video to audio: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_karaoke_from_youtube(
        self,
        youtube_url: str,
        output_path: str,
        srt_content: Optional[str] = None,
        logo_path: Optional[str] = None,
        auto_transcribe: bool = True,
        separate_vocals: bool = False,
        genius_lyrics: Optional[str] = None,
        vocals_file: Optional[str] = None,
        auto_detect_offset: bool = False,
        fast_mode: bool = False,
        video_file: Optional[str] = None,
        airtable_record_id: Optional[str] = None,
        audio_file: Optional[str] = None,
        audio_to_video_offset: float = 0.0,
        language: Optional[str] = None,
        **kwargs
    ) -> Tuple[bool, Optional[dict]]:
        """
        Generate a karaoke video from a YouTube URL.
        Downloads video, transcribes if needed, and generates karaoke.
        
        Args:
            youtube_url: YouTube video URL
            output_path: Output video file path
            srt_content: Optional pre-existing SRT content (skips transcription)
            logo_path: Optional logo file path
            auto_transcribe: If True and no srt_content, transcribe with Whisper
            separate_vocals: If True, separate vocals before transcription for better accuracy
            
        Returns:
            Tuple of (success, output_path or error_message)
        """
        temp_video = None
        try:
            # Store record_id for progressive saves
            self._airtable_record_id = airtable_record_id
            
            # Use provided video file or download from YouTube
            if video_file and os.path.exists(video_file):
                print(f"✓ Using provided video file: {video_file}")
                temp_video = video_file
            else:
                # Download video to temporary file
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    temp_video = tmp.name
                
                print(f"📥 Downloading video from YouTube...")
                if not self.download_youtube_video(youtube_url, temp_video):
                    return False, "Failed to download YouTube video"
                
                # Upload video to Airtable automatically
                if airtable_record_id:
                    video_size = os.path.getsize(temp_video) / (1024 * 1024)
                    print(f"💾 Video downloaded: {os.path.basename(temp_video)} ({video_size:.1f} MB)")
                    
                    # Upload to Airtable
                    from src.airtable_client import AirtableClient
                    airtable = AirtableClient()
                    if airtable.upload_video_file(temp_video, airtable_record_id):
                        print(f"✅ Video uploaded to Airtable")
                    else:
                        print(f"⚠️  Video upload failed, you may need to upload manually")
            
            # STEP 1: If we have original audio, sync video with it FIRST
            video_for_karaoke = temp_video
            if audio_file:
                print("🎵 Original audio detected - syncing video with audio FIRST...")
                if audio_to_video_offset != 0:
                    print(f"⏱️  Using manual offset: {audio_to_video_offset:+.1f}s")
                
                synced_video = tempfile.NamedTemporaryFile(suffix="_synced.mp4", delete=False).name
                
                if self.sync_video_to_audio(temp_video, audio_file, synced_video, fast_mode=fast_mode, manual_offset=audio_to_video_offset):
                    video_for_karaoke = synced_video
                    print("✅ Video synchronized with original audio")
                else:
                    print("⚠️  Sync failed, using original video")
                    audio_file = None  # Fallback to video audio
            
            # STEP 2: Get or generate SRT content (AFTER sync if audio_file exists)
            if not srt_content:
                if auto_transcribe:
                    # Priority: vocals_file > audio_file ONLY (NEVER use video for transcription)
                    if vocals_file:
                        print(f"🎵 Using pre-separated vocals for transcription: {vocals_file}")
                        audio_to_transcribe = vocals_file
                    elif audio_file:
                        print(f"🎵 Transcribing from original audio file: {audio_file}")
                        audio_to_transcribe = audio_file
                        # If we have original audio and need vocals, separate from it
                        if separate_vocals:
                            print("🎵 Will separate vocals from original audio for better transcription")
                    else:
                        print("❌ ERROR: No audio file or vocal file available for transcription!")
                        print("   REQUIREMENT: You must have either:")
                        print("   1. Original audio file (Google Drive link in Airtable)")
                        print("   2. Pre-separated vocal file (uploaded to Airtable)")
                        print("   Video files should NEVER be used for transcription (subtitle artifacts)")
                        return None
                    
                    if genius_lyrics:
                        print("🎵 Will sync Genius lyrics with Whisper timing")
                    
                    # Preserve original timecodes when using original audio
                    preserve_timecodes = audio_file is not None
                    
                    # Store record ID for language saving
                    if airtable_record_id:
                        self._airtable_record_id = airtable_record_id
                    
                    srt_content = self.transcribe_video(
                        audio_to_transcribe,
                        separate_vocals=separate_vocals if not vocals_file else False,
                        genius_lyrics=genius_lyrics,
                        preserve_timecodes=preserve_timecodes,
                        language=language
                    )
                    if not srt_content:
                        return False, "Failed to transcribe video"
                else:
                    return False, "No SRT content provided and auto_transcribe is False"
            
            # STEP 3: No offset needed when using original audio!
            # Since we transcribe from the original audio and use it in the final video,
            # the Whisper timecodes already match perfectly - no adjustment needed
            detected_offset = 0.0
            
            if audio_file:
                print("✅ Using original audio timecodes (no offset needed)")
            elif auto_detect_offset and srt_content:
                # Only detect offset if NOT using original audio (fallback case)
                print("🔍 Detecting singing start time...")
                try:
                    from src.audio_analyzer import AudioAnalyzer
                    analyzer = AudioAnalyzer()
                    
                    # Use the same audio source we transcribed from
                    audio_for_detection = vocals_file if vocals_file else video_for_karaoke
                    
                    detected_offset, confidence = analyzer.detect_intro_duration(
                        video_for_karaoke,
                        audio_for_detection if audio_for_detection != video_for_karaoke else None
                    )
                    
                    print(f"✓ Detected offset: {detected_offset:.1f}s (confidence: {confidence})")
                    
                    # Apply offset to SRT if detected
                    if detected_offset > 0 and srt_content:
                        print(f"⏱️  Applying detected offset: {detected_offset:.1f}s")
                        srt_content = self.adjust_srt_timing(srt_content, detected_offset)
                except Exception as e:
                    print(f"⚠️  Offset detection failed: {e}")
                    detected_offset = 0.0
            
            # Generate karaoke
            print("🎬 Generating karaoke video...")
            success, result = self.generate_karaoke(
                video_path=video_for_karaoke,
                srt_content=srt_content,
                output_path=output_path,
                logo_path=logo_path,
                fast_mode=fast_mode
            )
            
            if success:
                # Track vocal file path if it was generated
                vocal_path = None
                if 'audio_to_transcribe' in locals() and audio_to_transcribe != temp_video:
                    vocal_path = audio_to_transcribe
                elif vocals_file:
                    vocal_path = vocals_file
                
                return True, {
                    'video_path': result,
                    'srt_content': srt_content,
                    'detected_offset': detected_offset,
                    'vocal_path': vocal_path
                }
            
            return success, result
            
        except Exception as e:
            return False, str(e)
        finally:
            # Clean up temporary video file
            if temp_video and os.path.exists(temp_video):
                try:
                    os.unlink(temp_video)
                except:
                    pass
