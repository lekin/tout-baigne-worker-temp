"""
Musixmatch API integration for retrieving synced lyrics.
Supports both subtitle (LRC) and richsync (character-level) formats.
"""

import os
import json
import requests
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import re
import textwrap


@dataclass
class MusixmatchTrack:
    """Musixmatch track information."""
    track_id: int
    track_name: str
    artist_name: str
    album_name: str
    track_length: int  # Duration in seconds
    has_subtitles: bool
    has_richsync: bool
    has_lyrics: bool


@dataclass
class SyncedLine:
    """A single synced lyric line."""
    start: float
    end: float
    text: str


class MusixmatchClient:
    """Client for Musixmatch Lyrics API."""
    
    BASE_URL = "https://api.musixmatch.com/ws/1.1"
    
    def __init__(self, api_key: str):
        """Initialize Musixmatch client.
        
        Args:
            api_key: Your Musixmatch API key
        """
        self.api_key = api_key
        self.session = requests.Session()
    
    def _make_request(self, endpoint: str, params: Dict) -> Dict:
        """
        Make a request to the Musixmatch API.
        
        Args:
            endpoint: API endpoint (e.g., 'track.search')
            params: Query parameters
            
        Returns:
            Response JSON
        """
        params['apikey'] = self.api_key
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Check API status
            status_code = data.get('message', {}).get('header', {}).get('status_code')
            if status_code != 200:
                raise Exception(f"Musixmatch API error: status_code={status_code}")
            
            return data
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {e}")

    def _debug_enabled(self) -> bool:
        v = os.getenv('MUSIXMATCH_DEBUG')
        return bool(v) and v != '0'
    
    def search_track(
        self,
        track_name: str,
        artist_name: str,
        album_name: Optional[str] = None
    ) -> List[MusixmatchTrack]:
        """
        Search for a track in Musixmatch database.
        
        Args:
            track_name: Song title
            artist_name: Artist name
            album_name: Album name (optional)
            
        Returns:
            List of matching tracks
        """
        params = {
            'q_track': track_name,
            'q_artist': artist_name,
            'page_size': 25,
            'page': 1,
            's_track_rating': 'desc'  # Sort by popularity
        }
        
        if album_name:
            params['q_album'] = album_name
        
        print(f"🔍 Searching Musixmatch for: '{track_name}' by '{artist_name}'")
        
        data = self._make_request('track.search', params)
        track_list = data.get('message', {}).get('body', {}).get('track_list', [])
        
        results = []
        for item in track_list:
            track_data = item.get('track', {})
            results.append(MusixmatchTrack(
                track_id=track_data.get('track_id'),
                track_name=track_data.get('track_name', ''),
                artist_name=track_data.get('artist_name', ''),
                album_name=track_data.get('album_name', ''),
                track_length=track_data.get('track_length', 0),
                has_subtitles=bool(track_data.get('has_subtitles', 0)),
                has_richsync=bool(track_data.get('has_richsync', 0)),
                has_lyrics=bool(track_data.get('has_lyrics', 0))
            ))
        
        print(f"✅ Found {len(results)} matching tracks")
        return results

    def matcher_track_get(
        self,
        track_name: str,
        artist_name: str,
        album_name: Optional[str] = None,
        track_isrc: Optional[str] = None,
    ) -> Optional[MusixmatchTrack]:
        params: Dict = {
            'q_track': track_name,
            'q_artist': artist_name,
        }
        if album_name:
            params['q_album'] = album_name
        if track_isrc:
            params['track_isrc'] = track_isrc

        try:
            data = self._make_request('matcher.track.get', params)
            track_data = data.get('message', {}).get('body', {}).get('track')
            if not track_data:
                return None
            return MusixmatchTrack(
                track_id=track_data.get('track_id'),
                track_name=track_data.get('track_name', ''),
                artist_name=track_data.get('artist_name', ''),
                album_name=track_data.get('album_name', ''),
                track_length=track_data.get('track_length', 0),
                has_subtitles=bool(track_data.get('has_subtitles', 0)),
                has_richsync=bool(track_data.get('has_richsync', 0)),
                has_lyrics=bool(track_data.get('has_lyrics', 0))
            )
        except Exception as e:
            if self._debug_enabled():
                print(f"⚠️  matcher.track.get failed: {e}")
            return None
    
    def get_subtitle(
        self,
        track_id: int,
        subtitle_format: str = 'lrc',
        subtitle_length: Optional[int] = None
    ) -> Optional[str]:
        """
        Get synced lyrics (subtitle) for a track.
        
        Args:
            track_id: Musixmatch track ID
            subtitle_format: Format ('lrc', 'dfxp', or 'mxm' for JSON)
            subtitle_length: Desired length in seconds (optional)
            
        Returns:
            Subtitle content as string
        """
        params = {
            'track_id': track_id,
            'subtitle_format': subtitle_format
        }
        
        if subtitle_length:
            params['f_subtitle_length'] = subtitle_length
            params['f_subtitle_length_max_deviation'] = 5
        
        print(f"📥 Fetching subtitle for track_id={track_id} (format={subtitle_format})")
        
        try:
            data = self._make_request('track.subtitle.get', params)
            subtitle_data = data.get('message', {}).get('body', {}).get('subtitle', {})
            
            if not subtitle_data:
                print("⚠️  No subtitle data available")
                return None
            
            subtitle_body = subtitle_data.get('subtitle_body', '')
            
            if not subtitle_body:
                print("⚠️  Subtitle body is empty")
                return None
            
            print(f"✅ Retrieved subtitle ({len(subtitle_body)} chars)")
            return subtitle_body
            
        except Exception as e:
            print(f"❌ Failed to get subtitle: {e}")
            return None
    
    def get_richsync(
        self,
        track_id: int,
        sync_length: Optional[int] = None
    ) -> Optional[str]:
        """
        Get richsync (character-level sync) for a track.
        
        Args:
            track_id: Musixmatch track ID
            sync_length: Desired length in seconds (optional)
            
        Returns:
            Richsync JSON string
        """
        params = {
            'track_id': track_id
        }
        
        if sync_length:
            params['f_sync_length'] = sync_length
            params['f_sync_length_max_deviation'] = 5
        
        print(f"📥 Fetching richsync for track_id={track_id}")
        
        try:
            data = self._make_request('track.richsync.get', params)
            richsync_data = data.get('message', {}).get('body', {}).get('richsync', {})
            
            if not richsync_data:
                print("⚠️  No richsync data available")
                return None
            
            richsync_body = richsync_data.get('richsync_body', '')
            
            if not richsync_body:
                print("⚠️  Richsync body is empty")
                return None
            
            print(f"✅ Retrieved richsync ({len(richsync_body)} chars)")
            return richsync_body
            
        except Exception as e:
            print(f"❌ Failed to get richsync: {e}")
            return None

    def get_track_length(self, track_id: int) -> Optional[int]:
        """Fetch track length using track.get.

        Musixmatch may return track_length=0 for track.search results.
        This helper hydrates the actual duration.
        """
        try:
            data = self._make_request('track.get', {'track_id': track_id})
            track = data.get('message', {}).get('body', {}).get('track', {})
            length = track.get('track_length')
            if isinstance(length, (int, float)):
                return int(length)
            if isinstance(length, str):
                try:
                    return int(float(length))
                except (ValueError, TypeError):
                    return None
            return None
        except Exception as e:
            if self._debug_enabled():
                print(f"⚠️  track.get failed for track_id={track_id}: {e}")
            return None

    def hydrate_track_lengths(self, tracks: List[MusixmatchTrack], limit: Optional[int] = None) -> None:
        """In-place hydration of track_length for tracks where it's missing/zero."""
        items = tracks[:limit] if limit is not None else tracks
        for t in items:
            try:
                if t.track_length and int(t.track_length) > 0:
                    continue
            except Exception:
                pass

            length = self.get_track_length(int(t.track_id))
            if length and length > 0:
                t.track_length = int(length)


class LRCParser:
    """Parse LRC (Lyric) format to synced lines."""
    
    @staticmethod
    def parse(lrc_content: str) -> List[SyncedLine]:
        """
        Parse LRC format to synced lines.
        
        LRC format example:
        [00:12.00]Line 1
        [00:17.20]Line 2
        
        Args:
            lrc_content: LRC format string
            
        Returns:
            List of SyncedLine objects
        """
        lines = []
        lrc_lines = lrc_content.strip().split('\n')
        
        # Time tags can be repeated: [mm:ss.xx][mm:ss.xx]text
        # Fractions can be omitted or have 2-3 digits.
        time_tag = re.compile(r"\[(\d+):(\d+)(?:\.(\d+))?\]")
        metadata_tag = re.compile(r"^\[[A-Za-z]+:.*\]$")
        
        parsed_entries = []
        
        for raw_line in lrc_lines:
            line = raw_line.strip()
            if not line:
                continue
            if metadata_tag.match(line):
                continue

            matches = list(time_tag.finditer(line))
            if not matches:
                continue

            text = time_tag.sub("", line).strip()
            if not text:
                continue

            for m in matches:
                minutes = int(m.group(1))
                seconds = int(m.group(2))
                frac = m.group(3) or "0"
                if len(frac) >= 3:
                    fraction_seconds = int(frac[:3]) / 1000.0
                else:
                    fraction_seconds = int(frac) / 100.0
                timestamp = minutes * 60 + seconds + fraction_seconds
                parsed_entries.append((timestamp, text))
        
        # Sort by timestamp
        parsed_entries.sort(key=lambda x: x[0])
        
        # Create SyncedLine objects with end times
        for i, (start, text) in enumerate(parsed_entries):
            # End time is the start of the next line, or +3 seconds for the last line
            if i < len(parsed_entries) - 1:
                end = parsed_entries[i + 1][0]
            else:
                end = start + 3.0
            
            lines.append(SyncedLine(start=start, end=end, text=text))
        
        return lines


class RichsyncParser:
    """Parse Musixmatch richsync format to synced lines."""
    
    @staticmethod
    def parse(richsync_json: str) -> List[SyncedLine]:
        """
        Parse Musixmatch richsync JSON to synced lines.
        
        Richsync format is a JSON array with character-level timing.
        We'll aggregate by line.
        
        Args:
            richsync_json: Richsync JSON string
            
        Returns:
            List of SyncedLine objects
        """
        try:
            data = json.loads(richsync_json)
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse richsync JSON: {e}")
            return []
        
        if not isinstance(data, list):
            print("⚠️  Richsync data is not a list")
            return []
        
        lines = []
        
        for entry in data:
            if not isinstance(entry, dict):
                continue
            
            # Richsync format: {"ts": start_time, "te": end_time, "l": [{"c": "text", "o": offset}]}
            start = entry.get('ts', 0)
            end = entry.get('te', start + 3)
            
            # Extract text from character array
            text_parts = []
            char_list = entry.get('l', [])
            for char_entry in char_list:
                if isinstance(char_entry, dict):
                    text_parts.append(char_entry.get('c', ''))
            
            text = ''.join(text_parts).strip()
            
            if text:
                lines.append(SyncedLine(start=start, end=end, text=text))
        
        return lines


class ASSKaraokeGenerator:
    """Generate ASS karaoke format with word-level timing from Richsync."""

    @staticmethod
    def _cap_first_letter(text: str) -> str:
        s = text or ""
        for i, ch in enumerate(s):
            if ch.isalpha():
                return s[:i] + ch.upper() + s[i + 1 :]
        return s

    @staticmethod
    def _spotify_slot_y(video_height: int, visible_relative_indices: List[int], relative_index: int) -> int:
        line_step = max(92, int(video_height * 0.115))
        center_y = int(video_height / 2) - 100
        return int(round(center_y + (relative_index * line_step)))

    @staticmethod
    def _spotify_window_timing(
        item_starts: List[float],
        item_ends: List[float],
        index: int,
        *,
        tail_padding_seconds: float = 0.8,
        transition_seconds: float = 0.54,
    ) -> Tuple[float, float, float]:
        window_start = max(0.0, float(item_starts[index]))
        if index + 1 < len(item_starts):
            window_end = max(window_start + 0.2, float(item_starts[index + 1]))
            transition = min(
                transition_seconds,
                max(0.16, min(transition_seconds, (window_end - window_start) * 0.55)),
            )
            stable_end = max(window_start, window_end - transition)
        else:
            window_end = max(window_start + 0.2, float(item_ends[index]) + tail_padding_seconds)
            stable_end = window_end
        return window_start, stable_end, window_end

    @staticmethod
    def _spotify_motion_override(
        event_start: float,
        event_end: float,
        motion_start: float,
        center_x: int,
        y0: int,
        y1: int,
        extra_tags: str = "",
    ) -> str:
        if event_end <= event_start or y0 == y1:
            return f"{{\\an8\\pos({center_x},{y0}){extra_tags}}}"

        motion_anchor = max(event_start, min(motion_start, event_end))
        t1_ms = max(0, int(round((motion_anchor - event_start) * 1000.0)))
        t2_ms = max(t1_ms + 1, int(round((event_end - event_start) * 1000.0)))
        return f"{{\\an8\\move({center_x},{y0},{center_x},{y1},{t1_ms},{t2_ms}){extra_tags}}}"

    @staticmethod
    def _spotify_ease_smootherstep(t: float) -> float:
        t = max(0.0, min(1.0, t))
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    @staticmethod
    def _spotify_segment_alpha(
        segment_start: float,
        segment_end: float,
        transition_start: float,
        transition_end: float,
        alpha_start: int,
        alpha_end: int,
    ) -> str:
        if alpha_start == alpha_end:
            alpha = int(alpha_end)
        elif transition_end <= transition_start:
            alpha = int(alpha_end)
        elif segment_end <= transition_start:
            alpha = int(alpha_start)
        elif segment_start >= transition_end:
            alpha = int(alpha_end)
        else:
            midpoint = max(transition_start, min(transition_end, (segment_start + segment_end) * 0.5))
            progress = (midpoint - transition_start) / max(transition_end - transition_start, 1e-6)
            progress = ASSKaraokeGenerator._spotify_ease_smootherstep(progress)
            alpha = int(round(alpha_start + ((alpha_end - alpha_start) * progress)))
        alpha = max(0, min(255, int(alpha)))
        if alpha <= 0:
            return ""
        return f"\\alpha&H{alpha:02X}&"

    @staticmethod
    def _spotify_motion_segments(
        start_time: float,
        stable_end: float,
        end_time: float,
        y0: int,
        y1: int,
    ) -> List[Tuple[float, float, int, int]]:
        if end_time <= start_time:
            return []
        if y0 == y1 or stable_end >= end_time:
            return [(start_time, end_time, y0, y0)]

        segments: List[Tuple[float, float, int, int]] = []
        if stable_end > start_time:
            segments.append((start_time, stable_end, y0, y0))

        transition_start = max(start_time, min(stable_end, end_time))
        transition_duration = end_time - transition_start
        if transition_duration <= 0:
            return segments or [(start_time, end_time, y0, y0)]

        step_count = 24
        checkpoints: List[Tuple[float, float]] = []
        for step_idx in range(step_count + 1):
            t = float(step_idx) / float(step_count)
            checkpoints.append((t, ASSKaraokeGenerator._spotify_ease_smootherstep(t)))
        delta = y1 - y0

        raw_segments: List[Tuple[float, float, int, int]] = []
        for idx in range(len(checkpoints) - 1):
            t0f, p0 = checkpoints[idx]
            t1f, p1 = checkpoints[idx + 1]
            seg_start = transition_start + (transition_duration * t0f)
            seg_end = transition_start + (transition_duration * t1f)
            if seg_end <= seg_start:
                continue
            seg_y0 = int(round(y0 + (delta * p0)))
            seg_y1 = int(round(y0 + (delta * p1)))
            raw_segments.append((seg_start, seg_end, seg_y0, seg_y1))

        for seg_start, seg_end, seg_y0, seg_y1 in raw_segments:
            if not segments:
                segments.append((seg_start, seg_end, seg_y0, seg_y1))
                continue
            prev_start, prev_end, prev_y0, prev_y1 = segments[-1]
            if prev_y0 == prev_y1 == seg_y0 == seg_y1:
                segments[-1] = (prev_start, seg_end, prev_y0, seg_y1)
            else:
                segments.append((seg_start, seg_end, seg_y0, seg_y1))

        return segments or [(start_time, end_time, y0, y1)]

    @staticmethod
    def _extract_chorus_items(text: str) -> Tuple[str, List[Tuple[str, float, float]]]:
        raw = text or ""
        total_non_space = len(re.sub(r"\s+", "", raw))
        if total_non_space <= 0:
            return raw.strip(), []

        def _non_space_upto(idx: int) -> int:
            return len(re.sub(r"\s+", "", raw[: max(0, int(idx))]))

        items: List[Tuple[str, float, float]] = []
        for m in re.finditer(r"\(([^)]*)\)", raw):
            seg = (m.group(1) or "").strip()
            if not seg:
                continue
            before = _non_space_upto(m.start())
            inside = len(re.sub(r"\s+", "", seg))
            start_f = float(before) / float(total_non_space)
            end_f = float(before + max(inside, 1)) / float(total_non_space)
            start_f = max(0.0, min(1.0, start_f))
            end_f = max(0.0, min(1.0, end_f))
            if end_f <= start_f:
                end_f = min(1.0, start_f + 0.15)
            items.append((seg, start_f, end_f))

        main_text = re.sub(r"\([^)]*\)", " ", raw)
        main_text = re.sub(r"\s+", " ", main_text).strip()
        return main_text, items

    @staticmethod
    def _append_backing_vocal_events(
        ass_lines: List[str],
        chorus_text: str,
        chorus_start: float,
        chorus_end: float,
        hold_ceiling: float,
        display_floor: float,
        display_ceiling: float,
        video_width: int,
        video_height: int,
        outline_colour: str,
        shadow_style: str,
        main_style: str,
        layer_shadow: int = 0,
        layer_main: int = 1,
        backing_pre_roll_seconds: float = 0.25,
        backing_post_roll_seconds: float = 0.90,
        font_size_chorus: int = 190,
        fall_px: int = 160,
        appear_ms: int = 220,
        hold_extra: float = 2.0,
        stagger_cap: float = 0.35,
    ) -> None:
        chorus_words = re.findall(r"\S+", ASSKaraokeGenerator._cap_first_letter(chorus_text))
        if not chorus_words or chorus_end <= chorus_start:
            return

        base_y = int(video_height * 0.50)
        total = max(0.2, float(chorus_end - chorus_start))
        stagger = min(stagger_cap, (total / max(len(chorus_words), 1)) * 0.9)
        chorus_hold_end = min(hold_ceiling, chorus_end + hold_extra)

        avg_char_w = float(font_size_chorus) * 0.48
        space_w = avg_char_w * 0.9
        word_ws = [max(1.0, len(w) * avg_char_w) for w in chorus_words]
        total_w = sum(word_ws) + space_w * max(len(chorus_words) - 1, 0)
        available_w = float(video_width) * 0.92
        scale = min(1.0, available_w / max(total_w, 1.0))
        scale_pct = int(round(scale * 100.0))
        space_w_s = space_w * scale
        word_ws_s = [ww * scale for ww in word_ws]
        total_w_s = total_w * scale
        start_x = (float(video_width) / 2.0) - (total_w_s / 2.0)

        for wi, w in enumerate(chorus_words):
            ws = chorus_start + (wi * stagger)
            we = chorus_hold_end
            if we <= ws:
                continue

            ws_disp = max(display_floor, ws - backing_pre_roll_seconds)
            we_disp = min(display_ceiling, we + backing_post_roll_seconds)
            if we_disp <= ws_disp:
                continue

            dur_ms = int((we_disp - ws_disp) * 1000)
            fall_dur_ms = min(1200, max(600, int(dur_ms * 0.45)))
            fall_t1 = max(appear_ms, dur_ms - fall_dur_ms)
            fall_t2 = dur_ms
            fade_out_t1 = fall_t1
            fall_mid = int(fall_t1 + ((fall_t2 - fall_t1) * 0.5))

            x_center = start_x + (sum(word_ws_s[:wi]) + (space_w_s * wi) + (word_ws_s[wi] / 2.0))
            x = int(x_center)
            y0 = base_y
            y1 = base_y + fall_px
            y_mid = int(y0 + ((y1 - y0) * 0.65))

            ws_t = ASSKaraokeGenerator.format_ass_timestamp(ws_disp)
            we_t = ASSKaraokeGenerator.format_ass_timestamp(we_disp)

            chorus_shadow_fx = (
                f"{{\\an5\\bord0\\shad0"
                f"\\alpha&HFF&\\blur5\\fs{font_size_chorus}\\fscx{scale_pct}\\fscy{scale_pct}"
                f"\\1c&H000000&"
                f"\\t(0,{appear_ms},0.7,\\alpha&H44&)"
                f"\\pos({x},{y0})"
                f"\\t({fall_t1},{fall_mid},1.6,\\pos({x},{y_mid}))"
                f"\\t({fall_mid},{fall_t2},0.6,\\pos({x},{y1}))"
                f"\\3c&H000000&\\4c&H000000&"
                f"\\t({fade_out_t1},{fall_t2},1.4,\\alpha&HFF&\\blur5)}}"
            )
            chorus_main_fx = (
                f"{{\\an5\\bord0\\shad0"
                f"\\alpha&HFF&\\blur4\\fs{font_size_chorus}\\fscx{scale_pct}\\fscy{scale_pct}"
                f"\\1c{outline_colour}&"
                f"\\t(0,{appear_ms},0.7,\\alpha&H44&)"
                f"\\pos({x},{y0})"
                f"\\t({fall_t1},{fall_mid},1.6,\\pos({x},{y_mid}))"
                f"\\t({fall_mid},{fall_t2},0.6,\\pos({x},{y1}))"
                f"\\t({fade_out_t1},{fall_t2},1.4,\\alpha&HFF&\\blur4)}}"
            )

            ass_lines.append(f"Dialogue: {layer_shadow},{ws_t},{we_t},{shadow_style},,0,0,0,,{chorus_shadow_fx}{w}\n")
            ass_lines.append(f"Dialogue: {layer_main},{ws_t},{we_t},{main_style},,0,0,0,,{chorus_main_fx}{w}\n")

    @staticmethod
    def generate_from_lrc(
        lrc_content: str,
        video_width: int = 1920,
        video_height: int = 1080,
        font_path: str = "legacy/karaoke/fonts/SpaceMono-Regular.ttf",
        offset_seconds: float = 0.0,
        progressive_fill: bool = True,
        primary_colour: str = "&H0000FFFF",
        secondary_colour: str = "&H00CCCCCC",
        outline_colour: str = "&HA64DFF",
    ) -> str:
        lines = LRCParser.parse(lrc_content)
        return ASSKaraokeGenerator.generate_from_synced_lines(
            lines,
            video_width=video_width,
            video_height=video_height,
            font_path=font_path,
            offset_seconds=offset_seconds,
            progressive_fill=progressive_fill,
            primary_colour=primary_colour,
            secondary_colour=secondary_colour,
            outline_colour=outline_colour,
        )

    @staticmethod
    def generate_from_synced_lines_spotify(
        synced_lines: List[SyncedLine],
        video_width: int = 1920,
        video_height: int = 1080,
        font_path: str = "legacy/karaoke/fonts/SpaceMono-Regular.ttf",
        offset_seconds: float = 0.0,
        progressive_fill: bool = True,
        max_gap_seconds: float = 6.0,
        max_display_seconds: float = 4.0,
        min_display_seconds: float = 1.4,
        chars_per_second: float = 14.0,
        extra_padding_seconds: float = 0.6,
        primary_colour: str = "&H0000FFFF",
        secondary_colour: str = "&H00CCCCCC",
        outline_colour: str = "&HA64DFF",
    ) -> str:
        if not synced_lines:
            return ""

        region_width = int(video_width * 0.86)
        center_x = int(video_width / 2)
        max_chars = max(16, int(region_width / 34))
        karaoke_tag = "\\kf" if progressive_fill else "\\k"

        ass_content = f"""[Script Info]
Title: Spotify Lyrics Stack
ScriptType: v4.00+
Collisions: Normal
PlayResX: {video_width}
PlayResY: {video_height}
Timer: 100.0000

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: ContextPast,{font_path},76,{primary_colour},{primary_colour},{outline_colour},&H64000000,1,0,0,0,100,100,0,0,1,3,0,8,80,80,80,1
Style: ContextPastShadow,{font_path},76,&H00FFFFFF,&HFFFFFFFF,&HFFFFFF,&H00000000,1,0,0,0,100,100,0,0,1,0,0,8,80,80,80,1
Style: ContextFuture,{font_path},76,{secondary_colour},{secondary_colour},{outline_colour},&H64000000,1,0,0,0,100,100,0,0,1,3,0,8,80,80,80,1
Style: ContextFutureShadow,{font_path},76,&H00FFFFFF,&HFFFFFFFF,&HFFFFFF,&H00000000,1,0,0,0,100,100,0,0,1,0,0,8,80,80,80,1
Style: MainActive,{font_path},76,{primary_colour},{secondary_colour},{outline_colour},&H64000000,1,0,0,0,100,100,0,0,1,3,0,8,80,80,80,1
Style: ShadowActive,{font_path},76,&H00FFFFFF,&HFFFFFFFF,&HFFFFFF,&H64000000,1,0,0,0,100,100,0,0,1,0,0,8,80,80,80,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

        def _build_karaoke_text(text: str, duration_seconds: float) -> str:
            stripped = re.sub(r"\s+", " ", (text or "").strip())
            if not stripped:
                return ""

            stripped = ASSKaraokeGenerator._cap_first_letter(stripped)
            total_cs = max(int(duration_seconds * 100), 1)
            chunks = re.findall(r"\S+|\s+", stripped)
            non_space_chars = sum(len(c) for c in chunks if not c.isspace())
            if non_space_chars <= 0:
                return stripped

            max_segments = max(total_cs, 1)
            group_size = (non_space_chars + max_segments - 1) // max_segments
            group_size = max(group_size, 1)

            parts: List[str] = []
            for c in chunks:
                if c.isspace():
                    parts.append(c)
                    continue
                for i in range(0, len(c), group_size):
                    parts.append(c[i:i + group_size])

            timed_parts = [p for p in parts if not p.isspace()]
            if not timed_parts:
                return stripped

            base = max(total_cs // len(timed_parts), 1)
            durations = [base] * len(timed_parts)
            remainder = total_cs - (base * len(timed_parts))
            if remainder != 0:
                durations[-1] += remainder

            out = ""
            di = 0
            for p in parts:
                if p == "\n":
                    out += "\\N"
                    continue
                if p.isspace():
                    out += p
                    continue
                kcs = max(int(durations[di]), 1)
                di += 1
                out += "{" + karaoke_tag + str(kcs) + "}" + p
            return out

        def _clean_text(text: str) -> str:
            cleaned = re.sub(r"\s+", " ", (text or "").strip())
            cleaned = cleaned.replace("{", "(").replace("}", ")")
            if not cleaned:
                return ""
            wrapped = textwrap.wrap(
                ASSKaraokeGenerator._cap_first_letter(cleaned),
                width=max_chars,
                break_long_words=False,
                break_on_hyphens=False,
            )
            if len(wrapped) > 2:
                wrapped = [wrapped[0], " ".join(wrapped[1:])]
            if len(wrapped) == 2 and len(wrapped[1]) > max_chars:
                wrapped[1] = wrapped[1][: max(1, max_chars - 1)].rstrip() + "…"
            return "\\N".join(wrapped) if wrapped else ""

        items: List[Tuple[float, float, str]] = []
        for line in synced_lines:
            start = float(line.start) + offset_seconds
            natural_end = float(line.end) + offset_seconds
            if natural_end <= start:
                continue

            end = natural_end
            gap = natural_end - start
            if gap > max_gap_seconds:
                stripped = (line.text or "").strip()
                non_space_chars = len(re.sub(r"\s+", "", stripped))
                est = (non_space_chars / max(chars_per_second, 0.1)) + extra_padding_seconds
                est = max(min_display_seconds, min(max_display_seconds, est))
                end = min(natural_end, start + est)
            if end <= start:
                continue

            items.append((start, end, line.text))

        if not items:
            return ""

        lane_clearance = 0.04
        normalized_items: List[Tuple[float, float, str]] = []
        for idx, (start, end, text) in enumerate(items):
            clipped_end = end
            if idx + 1 < len(items):
                clipped_end = min(clipped_end, max(start + 0.12, items[idx + 1][0] - lane_clearance))
            normalized_items.append((start, clipped_end, text))
        items = normalized_items

        item_starts = [it[0] for it in items]
        item_ends = [it[1] for it in items]

        for active_index, (start, end, text_raw) in enumerate(items):
            window_start, stable_end, window_end = ASSKaraokeGenerator._spotify_window_timing(
                item_starts,
                item_ends,
                active_index,
            )
            window_start_t = ASSKaraokeGenerator.format_ass_timestamp(window_start)
            window_end_t = ASSKaraokeGenerator.format_ass_timestamp(window_end)
            move_start = stable_end
            current_visible_rels = [rel for rel in (-1, 0, 1) if 0 <= active_index + rel < len(items)]
            if active_index + 1 < len(items):
                next_visible_rels = [rel for rel in (-1, 0, 1) if 0 <= (active_index + 1) + rel < len(items)]
            else:
                next_visible_rels = list(current_visible_rels)

            main_text_raw, chorus_items = ASSKaraokeGenerator._extract_chorus_items(text_raw)
            karaoke_text = _build_karaoke_text(main_text_raw, end - start)
            active_text = _clean_text(main_text_raw)
            if not active_text:
                active_text = _clean_text(text_raw)
            if not karaoke_text and active_text:
                karaoke_text = active_text
            if chorus_items:
                backing_lines: List[str] = []
                duration_seconds = max(end - start, 0.01)
                for chorus_text, start_f, end_f in chorus_items:
                    chorus_start = start + (duration_seconds * start_f)
                    chorus_end = start + (duration_seconds * end_f) + 0.6
                    chorus_start = max(start, min(end, chorus_start))
                    chorus_end = max(chorus_start + 0.2, min(end, chorus_end))
                    if chorus_end <= chorus_start:
                        continue
                    ASSKaraokeGenerator._append_backing_vocal_events(
                        backing_lines,
                        chorus_text=chorus_text,
                        chorus_start=chorus_start,
                        chorus_end=chorus_end,
                        hold_ceiling=end,
                        display_floor=start,
                        display_ceiling=window_end,
                        video_width=video_width,
                        video_height=video_height,
                        outline_colour=outline_colour,
                        shadow_style="ShadowActive",
                        main_style="MainActive",
                        layer_shadow=0,
                        layer_main=1,
                    )
                if backing_lines:
                    ass_content += "".join(backing_lines)
            if not active_text:
                continue

            for rel in (-1, 1):
                line_index = active_index + rel
                if line_index < 0 or line_index >= len(items):
                    continue
                context_main_text, _ = ASSKaraokeGenerator._extract_chorus_items(items[line_index][2])
                context_text = _clean_text(context_main_text)
                if not context_text:
                    continue
                context_style = "ContextPast" if rel < 0 else "ContextFuture"
                context_shadow_style = "ContextPastShadow" if rel < 0 else "ContextFutureShadow"
                context_main_blur = "\\blur0.6"
                context_shadow_fx = "\\blur10\\3c&H000000&\\4c&H000000&" if rel < 0 else "\\blur2.2\\1c&HFFFFFF&\\3c&HFFFFFF&"

                y0 = ASSKaraokeGenerator._spotify_slot_y(video_height, current_visible_rels, rel)
                if active_index + 1 < len(items):
                    next_rel = line_index - (active_index + 1)
                    y1 = ASSKaraokeGenerator._spotify_slot_y(video_height, next_visible_rels, next_rel)
                else:
                    y1 = y0

                for seg_start, seg_end, seg_y0, seg_y1 in ASSKaraokeGenerator._spotify_motion_segments(
                    window_start,
                    move_start,
                    window_end,
                    y0,
                    y1,
                ):
                    seg_start_t = ASSKaraokeGenerator.format_ass_timestamp(seg_start)
                    seg_end_t = ASSKaraokeGenerator.format_ass_timestamp(seg_end)
                    fade_out_tag = ""
                    if rel < 0:
                        fade_out_tag = ASSKaraokeGenerator._spotify_segment_alpha(
                            seg_start,
                            seg_end,
                            move_start,
                            window_end,
                            0x00,
                            0xFF,
                        )
                    if seg_y0 == seg_y1:
                        shadow_motion = f"{{\\an8\\pos({center_x},{seg_y0}){context_shadow_fx}{fade_out_tag}}}"
                        main_motion = f"{{\\an8\\pos({center_x},{seg_y0}){context_main_blur}{fade_out_tag}}}"
                    else:
                        shadow_motion = f"{{\\an8\\move({center_x},{seg_y0},{center_x},{seg_y1}){context_shadow_fx}{fade_out_tag}}}"
                        main_motion = f"{{\\an8\\move({center_x},{seg_y0},{center_x},{seg_y1}){context_main_blur}{fade_out_tag}}}"
                    ass_content += f"Dialogue: 0,{seg_start_t},{seg_end_t},{context_shadow_style},,0,0,0,,{shadow_motion}{context_text}\n"
                    ass_content += f"Dialogue: 1,{seg_start_t},{seg_end_t},{context_style},,0,0,0,,{main_motion}{context_text}\n"

            incoming_index = active_index + 2
            if active_index + 1 < len(items) and incoming_index < len(items) and move_start < window_end:
                incoming_main_text, _ = ASSKaraokeGenerator._extract_chorus_items(items[incoming_index][2])
                incoming_text = _clean_text(incoming_main_text)
                if incoming_text:
                    incoming_y0 = ASSKaraokeGenerator._spotify_slot_y(video_height, current_visible_rels, 2)
                    incoming_y1 = ASSKaraokeGenerator._spotify_slot_y(video_height, next_visible_rels, 1)
                    incoming_fade_end = min(window_end, move_start + 0.32)
                    for seg_start, seg_end, seg_y0, seg_y1 in ASSKaraokeGenerator._spotify_motion_segments(
                        move_start,
                        move_start,
                        window_end,
                        incoming_y0,
                        incoming_y1,
                    ):
                        seg_start_t = ASSKaraokeGenerator.format_ass_timestamp(seg_start)
                        seg_end_t = ASSKaraokeGenerator.format_ass_timestamp(seg_end)
                        intro_alpha_tag = ASSKaraokeGenerator._spotify_segment_alpha(
                            seg_start,
                            seg_end,
                            move_start,
                            incoming_fade_end,
                            0xFF,
                            0x00,
                        )
                        if seg_y0 == seg_y1:
                            incoming_motion = f"{{\\an8\\pos({center_x},{seg_y0})\\blur0.6{intro_alpha_tag}}}"
                        else:
                            incoming_motion = f"{{\\an8\\move({center_x},{seg_y0},{center_x},{seg_y1})\\blur0.6{intro_alpha_tag}}}"
                        ass_content += f"Dialogue: 0,{seg_start_t},{seg_end_t},ContextFutureShadow,,0,0,0,,{incoming_motion}{incoming_text}\n"
                        ass_content += f"Dialogue: 1,{seg_start_t},{seg_end_t},ContextFuture,,0,0,0,,{incoming_motion}{incoming_text}\n"

            active_y0 = ASSKaraokeGenerator._spotify_slot_y(video_height, current_visible_rels, 0)
            active_fixed_end = max(window_start, min(move_start, window_end))
            if active_fixed_end > window_start:
                active_fixed_end_t = ASSKaraokeGenerator.format_ass_timestamp(active_fixed_end)
                fixed_shadow = f"{{\\an8\\pos({center_x},{active_y0})\\blur10\\3c&H000000&\\4c&H000000&}}"
                fixed_main = f"{{\\an8\\pos({center_x},{active_y0})}}"
                ass_content += f"Dialogue: 2,{window_start_t},{active_fixed_end_t},ShadowActive,,0,0,0,,{fixed_shadow}{karaoke_text}\n"
                ass_content += f"Dialogue: 3,{window_start_t},{active_fixed_end_t},MainActive,,0,0,0,,{fixed_main}{karaoke_text}\n"

            active_y1 = ASSKaraokeGenerator._spotify_slot_y(
                video_height,
                next_visible_rels,
                -1 if active_index + 1 < len(items) else 0,
            )
            if move_start < window_end:
                for seg_start, seg_end, seg_y0, seg_y1 in ASSKaraokeGenerator._spotify_motion_segments(
                    move_start,
                    move_start,
                    window_end,
                    active_y0,
                    active_y1,
                ):
                    seg_start_t = ASSKaraokeGenerator.format_ass_timestamp(seg_start)
                    seg_end_t = ASSKaraokeGenerator.format_ass_timestamp(seg_end)
                    if seg_y0 == seg_y1:
                        shadow_motion = f"{{\\an8\\pos({center_x},{seg_y0})\\blur10\\3c&H000000&\\4c&H000000&}}"
                        main_motion = f"{{\\an8\\pos({center_x},{seg_y0})}}"
                    else:
                        shadow_motion = f"{{\\an8\\move({center_x},{seg_y0},{center_x},{seg_y1})\\blur10\\3c&H000000&\\4c&H000000&}}"
                        main_motion = f"{{\\an8\\move({center_x},{seg_y0},{center_x},{seg_y1})}}"
                    ass_content += f"Dialogue: 2,{seg_start_t},{seg_end_t},ShadowActive,,0,0,0,,{shadow_motion}{active_text}\n"
                    ass_content += f"Dialogue: 3,{seg_start_t},{seg_end_t},MainActive,,0,0,0,,{main_motion}{active_text}\n"

        return ass_content

    @staticmethod
    def generate_from_richsync_spotify(
        richsync_json: str,
        video_width: int = 1920,
        video_height: int = 1080,
        font_path: str = "legacy/karaoke/fonts/SpaceMono-Regular.ttf",
        offset_seconds: float = 0.0,
        progressive_fill: bool = True,
        primary_colour: str = "&H0000FFFF",
        secondary_colour: str = "&H00CCCCCC",
        outline_colour: str = "&HA64DFF",
    ) -> str:
        try:
            data = json.loads(richsync_json)
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse richsync JSON: {e}")
            return ""

        if not isinstance(data, list):
            print("⚠️  Richsync data is not a list")
            return ""

        ass_content = f"""[Script Info]
Title: Karaoke Spotify Stack with Space Mono
ScriptType: v4.00+
Collisions: Normal
PlayResX: {video_width}
PlayResY: {video_height}
Timer: 100.0000

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: MainActive,{font_path},76,{primary_colour},{secondary_colour},{outline_colour},&H64000000,1,0,0,0,100,100,0,0,1,3,0,8,80,80,80,1
Style: ShadowActive,{font_path},76,&H00FFFFFF,&HFFFFFFFF,&HFFFFFF,&H64000000,1,0,0,0,100,100,0,0,1,0,0,8,80,80,80,1
Style: ContextPast,{font_path},76,{primary_colour},{primary_colour},{outline_colour},&H64000000,1,0,0,0,100,100,0,0,1,3,0,8,80,80,80,1
Style: ContextPastShadow,{font_path},76,&H00FFFFFF,&HFFFFFFFF,&HFFFFFF,&H00000000,1,0,0,0,100,100,0,0,1,0,0,8,80,80,80,1
Style: ContextFuture,{font_path},76,{secondary_colour},{secondary_colour},{outline_colour},&H64000000,1,0,0,0,100,100,0,0,1,3,0,8,80,80,80,1
Style: ContextFutureShadow,{font_path},76,&H00FFFFFF,&HFFFFFFFF,&HFFFFFF,&H00000000,1,0,0,0,100,100,0,0,1,0,0,8,80,80,80,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

        karaoke_tag = "\\kf" if progressive_fill else "\\k"
        lane_clearance = 0.04
        max_words_per_line = 6
        min_display_duration = 2.5
        overlap_duration = 0.5
        margin_l = 80
        margin_r = 80
        font_size = 76
        max_scale = 1.20
        avg_char_width = float(font_size) * 0.48 * max_scale
        available_w = max(1.0, float(video_width - margin_l - margin_r))
        max_chars_per_line = max(10, int(available_w / max(avg_char_width, 1.0)))
        center_x = int(video_width / 2)

        def _wrap_plain_text(text: str) -> str:
            text = re.sub(r"\s+", " ", (text or "").strip())
            text = text.replace("{", "(").replace("}", ")")
            if not text:
                return ""
            words = text.split()
            if len(words) <= max_words_per_line:
                return ASSKaraokeGenerator._cap_first_letter(text)
            left = " ".join(words[:max_words_per_line])
            right = " ".join(words[max_words_per_line:])
            if len(right) > max_chars_per_line:
                right = right[: max(1, max_chars_per_line - 1)].rstrip() + "…"
            return ASSKaraokeGenerator._cap_first_letter(left) + "\\N" + right

        chunks = []
        for entry in data:
            if not isinstance(entry, dict):
                continue

            line_start = entry.get('ts', 0)
            line_end = entry.get('te', line_start + 3)
            char_list = entry.get('l', [])
            if not char_list:
                continue

            words = []
            current_word = []
            for char_entry in char_list:
                if not isinstance(char_entry, dict):
                    continue
                text = char_entry.get('c', '')
                offset = char_entry.get('o', 0)
                if text.strip() == '':
                    if current_word:
                        words.append(current_word)
                        current_word = []
                else:
                    current_word.append({'text': text, 'offset': offset, 'entry': char_entry})
            if current_word:
                words.append(current_word)

            for chunk_idx in range(0, len(words), max_words_per_line):
                chunk_words = words[chunk_idx:chunk_idx + max_words_per_line]
                if not chunk_words:
                    continue

                chunk_start_offset = chunk_words[0][0]['offset']
                line_base = line_start + offset_seconds
                chunk_start = line_base + chunk_start_offset
                if chunk_idx + max_words_per_line < len(words):
                    natural_end_offset = words[chunk_idx + max_words_per_line][0]['offset']
                    natural_end = line_base + natural_end_offset
                else:
                    natural_end = line_end + offset_seconds

                natural_duration = natural_end - chunk_start
                display_duration = max(natural_duration, min_display_duration)
                chunk_end = chunk_start + display_duration + overlap_duration
                last_char_offset = chunk_words[-1][-1]['offset']
                sung_end = line_base + last_char_offset

                chunks.append({
                    'chunk_start': chunk_start,
                    'chunk_end': chunk_end,
                    'line_base': line_base,
                    'chunk_words': chunk_words,
                    'sung_end': sung_end,
                })

        for i in range(len(chunks) - 1):
            max_end = chunks[i + 1]['chunk_start'] - lane_clearance
            max_end = max(max_end, chunks[i]['chunk_start'] + 0.12)
            if chunks[i]['chunk_end'] > max_end:
                chunks[i]['chunk_end'] = max_end

        item_starts = [float(ch['chunk_start']) for ch in chunks]
        item_ends = [float(ch['chunk_end']) for ch in chunks]

        for line_counter, ch in enumerate(chunks):
            chunk_start = ch['chunk_start']
            chunk_end = ch['chunk_end']
            line_base = ch['line_base']
            chunk_words = ch['chunk_words']
            if chunk_end <= chunk_start:
                continue

            window_start, stable_end, window_end = ASSKaraokeGenerator._spotify_window_timing(
                item_starts,
                item_ends,
                line_counter,
            )
            window_start_t = ASSKaraokeGenerator.format_ass_timestamp(window_start)
            window_end_t = ASSKaraokeGenerator.format_ass_timestamp(window_end)
            move_start = stable_end
            current_visible_rels = [rel for rel in (-1, 0, 1) if 0 <= line_counter + rel < len(chunks)]
            if line_counter + 1 < len(chunks):
                next_visible_rels = [rel for rel in (-1, 0, 1) if 0 <= (line_counter + 1) + rel < len(chunks)]
            else:
                next_visible_rels = list(current_visible_rels)

            word_texts_all = ["".join((cd.get('text') or '') for cd in w) for w in chunk_words]
            chorus_start_offset: Optional[float] = None
            chorus_end_offset: Optional[float] = None
            chorus_text = ""
            chorus_split_idx: Optional[int] = None
            chorus_span_start: Optional[int] = None
            chorus_span_end: Optional[int] = None
            if word_texts_all:
                last_has_close = (')' in word_texts_all[-1])
                open_idx = None
                for i in range(len(word_texts_all) - 1, -1, -1):
                    if '(' in word_texts_all[i]:
                        open_idx = i
                        break
                if last_has_close and open_idx is not None and open_idx >= max(0, len(word_texts_all) - 8):
                    chorus_split_idx = int(open_idx)
                else:
                    open_any: Optional[int] = None
                    close_any: Optional[int] = None
                    for wi, wt in enumerate(word_texts_all):
                        if open_any is None and '(' in wt:
                            open_any = int(wi)
                        if open_any is not None and ')' in wt:
                            close_any = int(wi)
                    if open_any is not None and close_any is not None and close_any >= open_any:
                        chorus_words_span = chunk_words[open_any:close_any + 1]
                        if chorus_words_span:
                            chorus_span_start = int(open_any)
                            chorus_span_end = int(close_any)
                            chorus_start_offset = chorus_words_span[0][0]['offset']
                            chorus_end_offset = chorus_words_span[-1][-1]['offset']
                            chorus_text = " ".join(word_texts_all[open_any:close_any + 1])
                            chorus_text = chorus_text.replace('(', ' ').replace(')', ' ')
                            chorus_text = re.sub(r"\s+", " ", chorus_text).strip()
                            chorus_text = ASSKaraokeGenerator._cap_first_letter(chorus_text)

            main_chunk_words = chunk_words
            main_word_texts = word_texts_all
            if chorus_split_idx is not None and chorus_split_idx < len(chunk_words):
                chorus_words = chunk_words[chorus_split_idx:]
                main_chunk_words = chunk_words[:chorus_split_idx]
                main_word_texts = word_texts_all[:chorus_split_idx]
                if chorus_words:
                    chorus_start_offset = chorus_words[0][0]['offset']
                    chorus_end_offset = chorus_words[-1][-1]['offset']
                    chorus_text = " ".join(word_texts_all[chorus_split_idx:])
                    chorus_text = chorus_text.replace('(', ' ').replace(')', ' ')
                    chorus_text = re.sub(r"\s+", " ", chorus_text).strip()
                    chorus_text = ASSKaraokeGenerator._cap_first_letter(chorus_text)

            word_visible = [True for _ in range(len(main_chunk_words))]
            visible_indices = list(range(len(main_chunk_words)))
            visible_word_texts = list(main_word_texts)
            if chorus_split_idx is None and chorus_span_start is not None and chorus_span_end is not None:
                word_visible = [not (chorus_span_start <= wi <= chorus_span_end) for wi in range(len(chunk_words))]
                visible_indices = [wi for wi, vis in enumerate(word_visible) if vis]
                visible_word_texts = [word_texts_all[wi] for wi in visible_indices]

            total_non_space = sum(len(re.sub(r"\s+", "", t)) for t in visible_word_texts)
            wrap_after: Optional[int] = None
            if total_non_space > max_chars_per_line and len(main_word_texts) > 1:
                candidates = []
                running = 0
                for wi, wt in enumerate(visible_word_texts[:-1]):
                    running += len(re.sub(r"\s+", "", wt))
                    if running <= 0 or running >= total_non_space:
                        continue
                    score = max(running, total_non_space - running) * 100 + abs(running - (total_non_space - running))
                    candidates.append((score, wi))
                if candidates:
                    _, wrap_after_visible = min(candidates)
                    if 0 <= int(wrap_after_visible) < len(visible_indices):
                        wrap_after = int(visible_indices[int(wrap_after_visible)])

            if chorus_text and chorus_start_offset is not None and chorus_end_offset is not None:
                chorus_start = line_base + float(chorus_start_offset)
                chorus_end = min(chunk_end, (line_base + float(chorus_end_offset) + 0.6))
                if chorus_end > chorus_start:
                    backing_lines: List[str] = []
                    ASSKaraokeGenerator._append_backing_vocal_events(
                        backing_lines,
                        chorus_text=chorus_text,
                        chorus_start=chorus_start,
                        chorus_end=chorus_end,
                        hold_ceiling=chunk_end,
                        display_floor=chunk_start,
                        display_ceiling=window_end,
                        video_width=video_width,
                        video_height=video_height,
                        outline_colour=outline_colour,
                        shadow_style="ShadowActive",
                        main_style="MainActive",
                        layer_shadow=0,
                        layer_main=1,
                    )
                    if backing_lines:
                        ass_content += "".join(backing_lines)

            for rel in (-1, 1):
                line_index = line_counter + rel
                if line_index < 0 or line_index >= len(chunks):
                    continue
                ctx_words = ["".join((cd.get('text') or '') for cd in w) for w in chunks[line_index]['chunk_words']]
                ctx_main_text, _ = ASSKaraokeGenerator._extract_chorus_items(" ".join(ctx_words))
                plain_text = _wrap_plain_text(ctx_main_text)
                if not plain_text:
                    continue
                context_style = "ContextPast" if rel < 0 else "ContextFuture"
                context_shadow_style = "ContextPastShadow" if rel < 0 else "ContextFutureShadow"
                context_main_blur = "\\blur0.6"
                context_shadow_fx = "\\blur10\\3c&H000000&\\4c&H000000&" if rel < 0 else "\\blur2.2\\1c&HFFFFFF&\\3c&HFFFFFF&"
                y0 = ASSKaraokeGenerator._spotify_slot_y(video_height, current_visible_rels, rel)
                if line_counter + 1 < len(chunks):
                    next_rel = line_index - (line_counter + 1)
                    y1 = ASSKaraokeGenerator._spotify_slot_y(video_height, next_visible_rels, next_rel)
                else:
                    y1 = y0

                for seg_start, seg_end, seg_y0, seg_y1 in ASSKaraokeGenerator._spotify_motion_segments(
                    window_start,
                    move_start,
                    window_end,
                    y0,
                    y1,
                ):
                    seg_start_t = ASSKaraokeGenerator.format_ass_timestamp(seg_start)
                    seg_end_t = ASSKaraokeGenerator.format_ass_timestamp(seg_end)
                    fade_out_tag = ""
                    if rel < 0:
                        fade_out_tag = ASSKaraokeGenerator._spotify_segment_alpha(
                            seg_start,
                            seg_end,
                            move_start,
                            window_end,
                            0x00,
                            0xFF,
                        )
                    if seg_y0 == seg_y1:
                        context_motion = f"{{\\an8\\pos({center_x},{seg_y0}){context_main_blur}{fade_out_tag}}}"
                        shadow_motion = f"{{\\an8\\pos({center_x},{seg_y0}){context_shadow_fx}{fade_out_tag}}}"
                    else:
                        context_motion = f"{{\\an8\\move({center_x},{seg_y0},{center_x},{seg_y1}){context_main_blur}{fade_out_tag}}}"
                        shadow_motion = f"{{\\an8\\move({center_x},{seg_y0},{center_x},{seg_y1}){context_shadow_fx}{fade_out_tag}}}"
                    ass_content += f"Dialogue: 0,{seg_start_t},{seg_end_t},{context_shadow_style},,0,0,0,,{shadow_motion}{plain_text}\n"
                    ass_content += f"Dialogue: 1,{seg_start_t},{seg_end_t},{context_style},,0,0,0,,{context_motion}{plain_text}\n"

            incoming_index = line_counter + 2
            if line_counter + 1 < len(chunks) and incoming_index < len(chunks) and move_start < window_end:
                incoming_words = ["".join((cd.get('text') or '') for cd in w) for w in chunks[incoming_index]['chunk_words']]
                incoming_main_text, _ = ASSKaraokeGenerator._extract_chorus_items(" ".join(incoming_words))
                incoming_text = _wrap_plain_text(incoming_main_text)
                if incoming_text:
                    incoming_y0 = ASSKaraokeGenerator._spotify_slot_y(video_height, current_visible_rels, 2)
                    incoming_y1 = ASSKaraokeGenerator._spotify_slot_y(video_height, next_visible_rels, 1)
                    incoming_fade_end = min(window_end, move_start + 0.32)
                    for seg_start, seg_end, seg_y0, seg_y1 in ASSKaraokeGenerator._spotify_motion_segments(
                        move_start,
                        move_start,
                        window_end,
                        incoming_y0,
                        incoming_y1,
                    ):
                        seg_start_t = ASSKaraokeGenerator.format_ass_timestamp(seg_start)
                        seg_end_t = ASSKaraokeGenerator.format_ass_timestamp(seg_end)
                        intro_alpha_tag = ASSKaraokeGenerator._spotify_segment_alpha(
                            seg_start,
                            seg_end,
                            move_start,
                            incoming_fade_end,
                            0xFF,
                            0x00,
                        )
                        if seg_y0 == seg_y1:
                            incoming_motion = f"{{\\an8\\pos({center_x},{seg_y0})\\blur0.6{intro_alpha_tag}}}"
                        else:
                            incoming_motion = f"{{\\an8\\move({center_x},{seg_y0},{center_x},{seg_y1})\\blur0.6{intro_alpha_tag}}}"
                        ass_content += f"Dialogue: 0,{seg_start_t},{seg_end_t},ContextFutureShadow,,0,0,0,,{incoming_motion}{incoming_text}\n"
                        ass_content += f"Dialogue: 1,{seg_start_t},{seg_end_t},ContextFuture,,0,0,0,,{incoming_motion}{incoming_text}\n"

            karaoke_text = ""
            lead_cap_done = False
            effective_end_offset = min((chunk_end - line_base), (window_end - line_base))
            if chorus_split_idx is not None and chorus_start_offset is not None:
                effective_end_offset = min(float(effective_end_offset), float(chorus_start_offset))
            for word_idx, word in enumerate(main_chunk_words):
                for char_idx, char_data in enumerate(word):
                    text = char_data['text']
                    offset = char_data['offset']
                    if word_idx < len(word_visible) and not word_visible[word_idx]:
                        text = ""
                    if chorus_text and text in ("(", ")"):
                        text = ""
                    if not lead_cap_done and text and any(ch.isalpha() for ch in text):
                        text = ASSKaraokeGenerator._cap_first_letter(text)
                        lead_cap_done = True

                    if char_idx + 1 < len(word):
                        next_offset = word[char_idx + 1]['offset']
                    elif word_idx + 1 < len(main_chunk_words):
                        next_offset = main_chunk_words[word_idx + 1][0]['offset']
                    else:
                        next_offset = effective_end_offset

                    duration_cs = max(int(max(0.01, next_offset - offset) * 100), 1)
                    karaoke_text += "{" + karaoke_tag + str(duration_cs) + "}" + text

                if word_idx < len(main_chunk_words) - 1:
                    cur_vis = (word_idx < len(word_visible) and word_visible[word_idx])
                    if cur_vis and any(word_visible[j] for j in range(word_idx + 1, len(word_visible))):
                        if wrap_after is not None and word_idx == wrap_after:
                            karaoke_text += "\\N"
                        else:
                            karaoke_text += " "

            plain_active_text = _wrap_plain_text(" ".join(visible_word_texts))
            if not karaoke_text and plain_active_text:
                karaoke_text = plain_active_text
            if not karaoke_text:
                continue

            active_y0 = ASSKaraokeGenerator._spotify_slot_y(video_height, current_visible_rels, 0)
            active_fixed_end = max(window_start, min(move_start, window_end))
            if active_fixed_end > window_start:
                active_fixed_end_t = ASSKaraokeGenerator.format_ass_timestamp(active_fixed_end)
                fixed_shadow = f"{{\\an8\\pos({center_x},{active_y0})\\blur10\\3c&H000000&\\4c&H000000&}}"
                fixed_main = f"{{\\an8\\pos({center_x},{active_y0})}}"
                ass_content += f"Dialogue: 2,{window_start_t},{active_fixed_end_t},ShadowActive,,0,0,0,,{fixed_shadow}{karaoke_text}\n"
                ass_content += f"Dialogue: 3,{window_start_t},{active_fixed_end_t},MainActive,,0,0,0,,{fixed_main}{karaoke_text}\n"

            active_y1 = ASSKaraokeGenerator._spotify_slot_y(
                video_height,
                next_visible_rels,
                -1 if line_counter + 1 < len(chunks) else 0,
            )
            if move_start < window_end:
                if plain_active_text:
                    for seg_start, seg_end, seg_y0, seg_y1 in ASSKaraokeGenerator._spotify_motion_segments(
                        move_start,
                        move_start,
                        window_end,
                        active_y0,
                        active_y1,
                    ):
                        seg_start_t = ASSKaraokeGenerator.format_ass_timestamp(seg_start)
                        seg_end_t = ASSKaraokeGenerator.format_ass_timestamp(seg_end)
                        if seg_y0 == seg_y1:
                            shadow_effects = f"{{\\an8\\pos({center_x},{seg_y0})\\blur10\\3c&H000000&\\4c&H000000&}}"
                            main_effects = f"{{\\an8\\pos({center_x},{seg_y0})}}"
                        else:
                            shadow_effects = f"{{\\an8\\move({center_x},{seg_y0},{center_x},{seg_y1})\\blur10\\3c&H000000&\\4c&H000000&}}"
                            main_effects = f"{{\\an8\\move({center_x},{seg_y0},{center_x},{seg_y1})}}"
                        ass_content += f"Dialogue: 2,{seg_start_t},{seg_end_t},ShadowActive,,0,0,0,,{shadow_effects}{plain_active_text}\n"
                        ass_content += f"Dialogue: 3,{seg_start_t},{seg_end_t},MainActive,,0,0,0,,{main_effects}{plain_active_text}\n"

        return ass_content

    @staticmethod
    def generate_from_synced_lines(
        synced_lines: List[SyncedLine],
        video_width: int = 1920,
        video_height: int = 1080,
        font_path: str = "legacy/karaoke/fonts/SpaceMono-Regular.ttf",
        offset_seconds: float = 0.0,
        progressive_fill: bool = True,
        max_gap_seconds: float = 6.0,
        max_display_seconds: float = 4.0,
        min_display_seconds: float = 1.4,
        chars_per_second: float = 14.0,
        extra_padding_seconds: float = 0.6,
        primary_colour: str = "&H0000FFFF",
        secondary_colour: str = "&H00CCCCCC",
        outline_colour: str = "&HA64DFF",
    ) -> str:
        karaoke_tag = "\\kf" if progressive_fill else "\\k"

        layer_chorus_shadow = 0
        layer_chorus_main = 1
        layer_shadow = 2
        layer_main = 3

        margin_l = 80
        margin_r = 80
        font_size = 76
        max_scale = 1.20
        avg_char_width = float(font_size) * 0.48 * max_scale
        available_w = max(1.0, float(video_width - margin_l - margin_r))
        max_chars_per_line = max(10, int(available_w / max(avg_char_width, 1.0)))

        ass_content = f"""[Script Info]
Title: Karaoke with Space Mono
ScriptType: v4.00+
Collisions: Normal
PlayResX: {video_width}
PlayResY: {video_height}
Timer: 100.0000

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Main,{font_path},76,{primary_colour},{secondary_colour},{outline_colour},&H64000000,1,0,0,0,100,100,0,0,1,3,0,2,80,80,450,1
Style: Shadow,{font_path},76,&H00FFFFFF,&HFFFFFFFF,&HFFFFFF,&H64000000,1,0,0,0,100,100,0,0,1,0,0,2,80,80,450,1
Style: MainTop,{font_path},76,{primary_colour},{secondary_colour},{outline_colour},&H64000000,1,0,0,0,100,100,0,0,1,3,0,8,80,80,450,1
Style: ShadowTop,{font_path},76,&H00FFFFFF,&HFFFFFFFF,&HFFFFFF,&H64000000,1,0,0,0,100,100,0,0,1,0,0,8,80,80,450,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

        def _build_karaoke_text(text: str, duration_seconds: float) -> str:
            stripped = (text or "").strip()
            if not stripped:
                return ""

            stripped = ASSKaraokeGenerator._cap_first_letter(stripped)

            total_cs = max(int(duration_seconds * 100), 1)
            chunks = re.findall(r"\S+|\s+", stripped)

            non_space_chars = sum(len(c) for c in chunks if not c.isspace())
            if non_space_chars <= 0:
                return stripped

            max_segments = max(total_cs, 1)
            group_size = (non_space_chars + max_segments - 1) // max_segments
            group_size = max(group_size, 1)

            parts: List[str] = []
            for c in chunks:
                if c.isspace():
                    parts.append(c)
                    continue
                for i in range(0, len(c), group_size):
                    parts.append(c[i:i + group_size])

            timed_parts = [p for p in parts if not p.isspace()]
            if not timed_parts:
                return stripped

            base = total_cs // len(timed_parts)
            base = max(base, 1)
            durations = [base] * len(timed_parts)
            remainder = total_cs - (base * len(timed_parts))
            if remainder != 0:
                durations[-1] += remainder

            out = ""
            di = 0
            for p in parts:
                if p == "\n":
                    out += "\\N"
                    continue
                if p.isspace():
                    out += p
                    continue
                kcs = max(int(durations[di]), 1)
                di += 1
                out += "{" + karaoke_tag + str(kcs) + "}" + p
            return out

        def _split_lead_blocks(text: str) -> List[str]:
            stripped = (text or "").strip()
            if not stripped:
                return []

            stripped = ASSKaraokeGenerator._cap_first_letter(stripped)

            chunks = re.findall(r"\S+|\s+", stripped)
            total_non_space = sum(len(c) for c in chunks if not c.isspace())
            if total_non_space <= max_chars_per_line:
                return [stripped]

            candidates = []
            running = 0
            for i, tok in enumerate(chunks):
                if tok.isspace():
                    if running <= 0 or running >= total_non_space:
                        continue
                    score = max(running, total_non_space - running) * 100 + abs(running - (total_non_space - running))
                    candidates.append((score, i))
                else:
                    running += len(tok)

            if not candidates:
                return [stripped]

            _, best_idx = min(candidates)
            left = "".join(chunks[:best_idx]).strip()
            right = "".join(chunks[best_idx + 1 :]).strip()
            left = re.sub(r"\s+", " ", left)
            right = re.sub(r"\s+", " ", right)

            out: List[str] = []
            if left:
                out.append(left)
            if right:
                out.append(right)
            return out or [stripped]

        def _extract_chorus_items(text: str) -> Tuple[str, List[Tuple[str, float, float]]]:
            raw = text or ""
            total_non_space = len(re.sub(r"\s+", "", raw))
            if total_non_space <= 0:
                return raw.strip(), []

            def _non_space_upto(idx: int) -> int:
                return len(re.sub(r"\s+", "", raw[: max(0, int(idx))]))

            items: List[Tuple[str, float, float]] = []
            for m in re.finditer(r"\(([^)]*)\)", raw):
                seg = (m.group(1) or "").strip()
                if not seg:
                    continue
                before = _non_space_upto(m.start())
                inside = len(re.sub(r"\s+", "", seg))
                start_f = float(before) / float(total_non_space)
                end_f = float(before + max(inside, 1)) / float(total_non_space)
                start_f = max(0.0, min(1.0, start_f))
                end_f = max(0.0, min(1.0, end_f))
                if end_f <= start_f:
                    end_f = min(1.0, start_f + 0.15)
                items.append((seg, start_f, end_f))

            main_text = re.sub(r"\([^)]*\)", " ", raw)
            main_text = re.sub(r"\s+", " ", main_text).strip()
            return main_text, items

        for idx, line in enumerate(synced_lines):
            start = float(line.start) + offset_seconds
            natural_end = float(line.end) + offset_seconds
            if natural_end <= start:
                continue

            backing_pre_roll_seconds = 0.25
            backing_post_roll_seconds = 0.90

            gap = natural_end - start
            end = natural_end
            if gap > max_gap_seconds:
                stripped = (line.text or "").strip()
                non_space_chars = len(re.sub(r"\s+", "", stripped))
                est = (non_space_chars / max(chars_per_second, 0.1)) + extra_padding_seconds
                est = max(min_display_seconds, min(max_display_seconds, est))
                end = min(natural_end, start + est)
                if end <= start:
                    continue

            duration_seconds = end - start
            main_text, chorus_items = _extract_chorus_items(line.text)
            lead_blocks = _split_lead_blocks(main_text)

            if not lead_blocks and not chorus_items:
                continue

            is_top = (idx % 2 == 1)
            shadow_style = "ShadowTop" if is_top else "Shadow"
            main_style = "MainTop" if is_top else "Main"

            if chorus_items:
                for chorus_text, start_f, end_f in chorus_items:
                    chorus_start = start + (duration_seconds * start_f)
                    chorus_end = start + (duration_seconds * end_f) + 0.6
                    chorus_start = max(start, min(end, chorus_start))
                    chorus_end = max(chorus_start + 0.2, min(end, chorus_end))
                    if chorus_end <= chorus_start:
                        continue

                    chorus_words = re.findall(r"\S+", ASSKaraokeGenerator._cap_first_letter(chorus_text))
                    if not chorus_words:
                        continue

                    base_y = int(video_height * 0.50)
                    fall_px = 160

                    total = max(0.2, float(chorus_end - chorus_start))
                    stagger = min(0.35, (total / max(len(chorus_words), 1)) * 0.9)
                    hold_extra = 2.0
                    chorus_hold_end = min(natural_end, chorus_end + hold_extra)

                    font_size_chorus = 190
                    avg_char_w = float(font_size_chorus) * 0.48
                    space_w = avg_char_w * 0.9
                    word_ws = [max(1.0, len(w) * avg_char_w) for w in chorus_words]
                    total_w = sum(word_ws) + space_w * max(len(chorus_words) - 1, 0)
                    available_w = float(video_width) * 0.92
                    scale = min(1.0, available_w / max(total_w, 1.0))
                    scale_pct = int(round(scale * 100.0))
                    space_w_s = space_w * scale
                    word_ws_s = [ww * scale for ww in word_ws]
                    total_w_s = total_w * scale
                    start_x = (float(video_width) / 2.0) - (total_w_s / 2.0)

                    for wi, w in enumerate(chorus_words):
                        ws = chorus_start + (wi * stagger)
                        we = chorus_hold_end
                        if we <= ws:
                            continue

                        ws_disp = max(start, ws - backing_pre_roll_seconds)
                        upper_end = natural_end + backing_post_roll_seconds
                        if idx + 1 < len(synced_lines):
                            upper_end = min(
                                upper_end,
                                (float(synced_lines[idx + 1].start) + offset_seconds) - 0.05,
                            )
                        we_disp = min(upper_end, we + backing_post_roll_seconds)
                        if we_disp <= ws_disp:
                            continue

                        dur_ms = int((we_disp - ws_disp) * 1000)
                        appear_ms = 220
                        fall_dur_ms = min(1200, max(600, int(dur_ms * 0.45)))
                        fall_t1 = max(appear_ms, dur_ms - fall_dur_ms)
                        fall_t2 = dur_ms
                        fade_out_t1 = fall_t1
                        fall_mid = int(fall_t1 + ((fall_t2 - fall_t1) * 0.5))

                        x_center = start_x + (sum(word_ws_s[:wi]) + (space_w_s * wi) + (word_ws_s[wi] / 2.0))
                        x = int(x_center)
                        y0 = base_y
                        y1 = base_y + fall_px
                        y_mid = int(y0 + ((y1 - y0) * 0.65))

                        ws_t = ASSKaraokeGenerator.format_ass_timestamp(ws_disp)
                        we_t = ASSKaraokeGenerator.format_ass_timestamp(we_disp)

                        chorus_shadow_fx = (
                            f"{{\\an5\\bord0\\shad0"
                            f"\\alpha&HFF&\\blur5\\fs190\\fscx{scale_pct}\\fscy{scale_pct}"
                            f"\\1c&H000000&"
                            f"\\t(0,{appear_ms},0.7,\\alpha&H44&)"
                            f"\\pos({x},{y0})"
                            f"\\t({fall_t1},{fall_mid},1.6,\\pos({x},{y_mid}))"
                            f"\\t({fall_mid},{fall_t2},0.6,\\pos({x},{y1}))"
                            f"\\3c&H000000&\\4c&H000000&"
                            f"\\t({fade_out_t1},{fall_t2},1.4,\\alpha&HFF&\\blur5)}}"
                        )
                        chorus_main_fx = (
                            f"{{\\an5\\bord0\\shad0"
                            f"\\alpha&HFF&\\blur4\\fs190\\fscx{scale_pct}\\fscy{scale_pct}"
                            f"\\1c{outline_colour}&"
                            f"\\t(0,{appear_ms},0.7,\\alpha&H44&)"
                            f"\\pos({x},{y0})"
                            f"\\t({fall_t1},{fall_mid},1.6,\\pos({x},{y_mid}))"
                            f"\\t({fall_mid},{fall_t2},0.6,\\pos({x},{y1}))"
                            f"\\t({fade_out_t1},{fall_t2},1.4,\\alpha&HFF&\\blur4)}}"
                        )

                        ass_content += f"Dialogue: {layer_chorus_shadow},{ws_t},{we_t},{shadow_style},,0,0,0,,{chorus_shadow_fx}{w}\n"
                        ass_content += f"Dialogue: {layer_chorus_main},{ws_t},{we_t},{main_style},,0,0,0,,{chorus_main_fx}{w}\n"
            if lead_blocks:
                weights = [len(re.sub(r"\s+", "", b)) for b in lead_blocks]
                total_w = sum(weights)
                if total_w <= 0:
                    weights = [1 for _ in lead_blocks]
                    total_w = sum(weights)

                seg_start = start
                for bi, block in enumerate(lead_blocks):
                    if bi >= len(lead_blocks) - 1:
                        seg_end = end
                    else:
                        seg_end = seg_start + (duration_seconds * (float(weights[bi]) / float(total_w)))
                        seg_end = max(seg_start + 0.6, min(end, seg_end))

                    if seg_end <= seg_start:
                        continue

                    seg_dur = seg_end - seg_start
                    seg_ms = int(seg_dur * 1000)
                    seg_start_time = ASSKaraokeGenerator.format_ass_timestamp(seg_start)
                    seg_end_time = ASSKaraokeGenerator.format_ass_timestamp(seg_end)

                    karaoke_text = _build_karaoke_text(block, seg_dur)
                    if karaoke_text:
                        shadow_effects = f"{{\\fad(500,500)\\blur10\\3c&H000000&\\4c&H000000&\\fscx100\\fscy100\\t(0,{seg_ms},\\fscx120\\fscy120)}}"
                        ass_content += f"Dialogue: {layer_shadow},{seg_start_time},{seg_end_time},{shadow_style},,0,0,0,,{shadow_effects}{karaoke_text}\n"

                        main_effects = f"{{\\fad(500,500)\\fscx100\\fscy100\\t(0,{seg_ms},\\fscx120\\fscy120)}}"
                        ass_content += f"Dialogue: {layer_main},{seg_start_time},{seg_end_time},{main_style},,0,0,0,,{main_effects}{karaoke_text}\n"

                    seg_start = seg_end

        return ass_content

    @staticmethod
    def generate_from_synced_lines_reel(
        synced_lines: List[SyncedLine],
        video_width: int = 1080,
        video_height: int = 1920,
        font_path: str = "legacy/karaoke/fonts/SpaceMono-Regular.ttf",
        offset_seconds: float = 0.0,
        region_x: int = 70,
        region_y: int = 520,
        region_w: int = 946,
        region_h: int = 320,
        progressive_fill: bool = True,
        max_gap_seconds: float = 6.0,
        max_display_seconds: float = 4.0,
        min_display_seconds: float = 1.4,
        chars_per_second: float = 14.0,
        extra_padding_seconds: float = 0.6,
        primary_colour: str = "&H0000FFFF",
        secondary_colour: str = "&H00CCCCCC",
        outline_colour: str = "&HA64DFF",
    ) -> str:
        karaoke_tag = "\\kf" if progressive_fill else "\\k"

        layer_chorus_shadow = 0
        layer_chorus_main = 1
        layer_shadow = 2
        layer_main = 3

        font_size = 76
        max_scale = 1.04
        avg_char_width = float(font_size) * 0.48 * max_scale
        available_w = max(1.0, float(region_w))
        max_chars_per_line = max(10, int(available_w / max(avg_char_width, 1.0)))

        margin_l = region_x
        margin_r = max(0, video_width - (region_x + region_w))
        margin_v = region_y + region_h // 2 - 40

        ass_content = f"""[Script Info]
Title: Karaoke Reel with Space Mono
ScriptType: v4.00+
Collisions: Normal
PlayResX: {video_width}
PlayResY: {video_height}
Timer: 100.0000

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Main,{font_path},76,{primary_colour},{secondary_colour},{outline_colour},&H64000000,1,0,0,0,100,100,0,0,1,3,0,8,{margin_l},{margin_r},{margin_v},1
Style: Shadow,{font_path},76,&H00FFFFFF,&HFFFFFFFF,&HFFFFFF,&H64000000,1,0,0,0,100,100,0,0,1,0,0,8,{margin_l},{margin_r},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

        def _build_karaoke_text(text: str, duration_seconds: float) -> str:
            stripped = (text or "").strip()
            if not stripped:
                return ""

            stripped = ASSKaraokeGenerator._cap_first_letter(stripped)

            total_cs = max(int(duration_seconds * 100), 1)
            chunks = re.findall(r"\S+|\s+", stripped)

            non_space_chars = sum(len(c) for c in chunks if not c.isspace())
            if non_space_chars <= 0:
                return stripped

            max_segments = max(total_cs, 1)
            group_size = (non_space_chars + max_segments - 1) // max_segments
            group_size = max(group_size, 1)

            parts: List[str] = []
            for c in chunks:
                if c.isspace():
                    parts.append(c)
                    continue
                for i in range(0, len(c), group_size):
                    parts.append(c[i:i + group_size])

            timed_parts = [p for p in parts if not p.isspace()]
            if not timed_parts:
                return stripped

            base = total_cs // len(timed_parts)
            base = max(base, 1)
            durations = [base] * len(timed_parts)
            remainder = total_cs - (base * len(timed_parts))
            if remainder != 0:
                durations[-1] += remainder

            out = ""
            di = 0
            for p in parts:
                if p == "\n":
                    out += "\\N"
                    continue
                if p.isspace():
                    out += p
                    continue
                kcs = max(int(durations[di]), 1)
                di += 1
                out += "{" + karaoke_tag + str(kcs) + "}" + p
            return out

        def _split_lead_blocks(text: str) -> List[str]:
            stripped = (text or "").strip()
            if not stripped:
                return []

            stripped = ASSKaraokeGenerator._cap_first_letter(stripped)

            chunks = re.findall(r"\S+|\s+", stripped)
            total_non_space = sum(len(c) for c in chunks if not c.isspace())
            if total_non_space <= max_chars_per_line:
                return [stripped]

            candidates = []
            running = 0
            for i, tok in enumerate(chunks):
                if tok.isspace():
                    if running <= 0 or running >= total_non_space:
                        continue
                    score = max(running, total_non_space - running) * 100 + abs(running - (total_non_space - running))
                    candidates.append((score, i))
                else:
                    running += len(tok)

            if not candidates:
                return [stripped]

            _, best_idx = min(candidates)
            left = "".join(chunks[:best_idx]).strip()
            right = "".join(chunks[best_idx + 1 :]).strip()
            left = re.sub(r"\s+", " ", left)
            right = re.sub(r"\s+", " ", right)

            out: List[str] = []
            if left:
                out.append(left)
            if right:
                out.append(right)
            return out or [stripped]

        def _extract_chorus_items(text: str) -> Tuple[str, List[Tuple[str, float, float]]]:
            raw = text or ""
            total_non_space = len(re.sub(r"\s+", "", raw))
            if total_non_space <= 0:
                return raw.strip(), []

            def _non_space_upto(idx: int) -> int:
                return len(re.sub(r"\s+", "", raw[: max(0, int(idx))]))

            items: List[Tuple[str, float, float]] = []
            for m in re.finditer(r"\(([^)]*)\)", raw):
                seg = (m.group(1) or "").strip()
                if not seg:
                    continue
                before = _non_space_upto(m.start())
                inside = len(re.sub(r"\s+", "", seg))
                start_f = float(before) / float(total_non_space)
                end_f = float(before + max(inside, 1)) / float(total_non_space)
                start_f = max(0.0, min(1.0, start_f))
                end_f = max(0.0, min(1.0, end_f))
                if end_f <= start_f:
                    end_f = min(1.0, start_f + 0.15)
                items.append((seg, start_f, end_f))

            main_text = re.sub(r"\([^)]*\)", " ", raw)
            main_text = re.sub(r"\s+", " ", main_text).strip()
            return main_text, items

        for idx, line in enumerate(synced_lines):
            start = float(line.start) + offset_seconds
            natural_end = float(line.end) + offset_seconds
            if natural_end <= start:
                continue

            backing_pre_roll_seconds = 0.25
            backing_post_roll_seconds = 0.90

            gap = natural_end - start
            end = natural_end
            if gap > max_gap_seconds:
                stripped = (line.text or "").strip()
                non_space_chars = len(re.sub(r"\s+", "", stripped))
                est = (non_space_chars / max(chars_per_second, 0.1)) + extra_padding_seconds
                est = max(min_display_seconds, min(max_display_seconds, est))
                end = min(natural_end, start + est)
                if end <= start:
                    continue

            duration_seconds = end - start
            main_text, chorus_items = _extract_chorus_items(line.text)
            lead_blocks = _split_lead_blocks(main_text)
            if not lead_blocks and not chorus_items:
                continue

            shadow_effects = None
            main_effects = None

            if chorus_items:
                for chorus_text, start_f, end_f in chorus_items:
                    chorus_start = start + (duration_seconds * start_f)
                    chorus_end = start + (duration_seconds * end_f) + 0.6
                    chorus_start = max(start, min(end, chorus_start))
                    chorus_end = max(chorus_start + 0.2, min(end, chorus_end))
                    if chorus_end <= chorus_start:
                        continue

                    chorus_words = re.findall(r"\S+", ASSKaraokeGenerator._cap_first_letter(chorus_text))
                    if not chorus_words:
                        continue

                    base_y = int(region_y + (region_h * 0.50))
                    fall_px = 140

                    total = max(0.2, float(chorus_end - chorus_start))
                    stagger = min(0.32, (total / max(len(chorus_words), 1)) * 0.9)
                    hold_extra = 1.8
                    chorus_hold_end = min(natural_end, chorus_end + hold_extra)

                    font_size_chorus = 180
                    avg_char_w = float(font_size_chorus) * 0.48
                    space_w = avg_char_w * 0.9
                    word_ws = [max(1.0, len(w) * avg_char_w) for w in chorus_words]
                    total_w = sum(word_ws) + space_w * max(len(chorus_words) - 1, 0)
                    available_w = float(video_width) * 0.92
                    scale = min(1.0, available_w / max(total_w, 1.0))
                    scale_pct = int(round(scale * 100.0))
                    space_w_s = space_w * scale
                    word_ws_s = [ww * scale for ww in word_ws]
                    total_w_s = total_w * scale
                    start_x = (float(video_width) / 2.0) - (total_w_s / 2.0)

                    for wi, w in enumerate(chorus_words):
                        ws = chorus_start + (wi * stagger)
                        we = chorus_hold_end
                        if we <= ws:
                            continue

                        ws_disp = max(start, ws - backing_pre_roll_seconds)
                        upper_end = natural_end + backing_post_roll_seconds
                        if idx + 1 < len(synced_lines):
                            upper_end = min(
                                upper_end,
                                (float(synced_lines[idx + 1].start) + offset_seconds) - 0.05,
                            )
                        we_disp = min(upper_end, we + backing_post_roll_seconds)
                        if we_disp <= ws_disp:
                            continue

                        dur_ms = int((we_disp - ws_disp) * 1000)
                        appear_ms = 200
                        fall_dur_ms = min(1100, max(550, int(dur_ms * 0.45)))
                        fall_t1 = max(appear_ms, dur_ms - fall_dur_ms)
                        fall_t2 = dur_ms
                        fade_out_t1 = fall_t1
                        fall_mid = int(fall_t1 + ((fall_t2 - fall_t1) * 0.5))

                        x_center = start_x + (sum(word_ws_s[:wi]) + (space_w_s * wi) + (word_ws_s[wi] / 2.0))
                        x = int(x_center)
                        y0 = base_y
                        y1 = base_y + fall_px

                        y_mid = int(y0 + ((y1 - y0) * 0.65))

                        ws_t = ASSKaraokeGenerator.format_ass_timestamp(ws_disp)
                        we_t = ASSKaraokeGenerator.format_ass_timestamp(we_disp)

                        chorus_shadow_fx = (
                            f"{{\\an5\\bord0\\shad0"
                            f"\\alpha&HFF&\\blur5\\fs180\\fscx{scale_pct}\\fscy{scale_pct}"
                            f"\\1c&H000000&"
                            f"\\t(0,{appear_ms},0.7,\\alpha&H44&)"
                            f"\\pos({x},{y0})"
                            f"\\t({fall_t1},{fall_mid},1.6,\\pos({x},{y_mid}))"
                            f"\\t({fall_mid},{fall_t2},0.6,\\pos({x},{y1}))"
                            f"\\3c&H000000&\\4c&H000000&"
                            f"\\t({fade_out_t1},{fall_t2},1.4,\\alpha&HFF&\\blur5)}}"
                        )
                        chorus_main_fx = (
                            f"{{\\an5\\bord0\\shad0"
                            f"\\alpha&HFF&\\blur4\\fs180\\fscx{scale_pct}\\fscy{scale_pct}"
                            f"\\1c{outline_colour}&"
                            f"\\t(0,{appear_ms},0.7,\\alpha&H44&)"
                            f"\\pos({x},{y0})"
                            f"\\t({fall_t1},{fall_mid},1.6,\\pos({x},{y_mid}))"
                            f"\\t({fall_mid},{fall_t2},0.6,\\pos({x},{y1}))"
                            f"\\t({fade_out_t1},{fall_t2},1.4,\\alpha&HFF&\\blur4)}}"
                        )

                        ass_content += (
                            f"Dialogue: {layer_chorus_shadow},{ws_t},{we_t},Shadow,,0,0,0,,"
                            f"{chorus_shadow_fx}{w}\n"
                        )
                        ass_content += (
                            f"Dialogue: {layer_chorus_main},{ws_t},{we_t},Main,,0,0,0,,"
                            f"{chorus_main_fx}{w}\n"
                        )

            if lead_blocks:
                weights = [len(re.sub(r"\s+", "", b)) for b in lead_blocks]
                total_w = sum(weights)
                if total_w <= 0:
                    weights = [1 for _ in lead_blocks]
                    total_w = sum(weights)

                seg_start = start
                for bi, block in enumerate(lead_blocks):
                    if bi >= len(lead_blocks) - 1:
                        seg_end = end
                    else:
                        seg_end = seg_start + (duration_seconds * (float(weights[bi]) / float(total_w)))
                        seg_end = max(seg_start + 0.6, min(end, seg_end))

                    if seg_end <= seg_start:
                        continue

                    seg_dur = seg_end - seg_start
                    seg_ms = int(seg_dur * 1000)
                    seg_start_time = ASSKaraokeGenerator.format_ass_timestamp(seg_start)
                    seg_end_time = ASSKaraokeGenerator.format_ass_timestamp(seg_end)

                    karaoke_text = _build_karaoke_text(block, seg_dur)
                    if karaoke_text:
                        shadow_effects = (
                            f"{{\\fad(200,400)\\blur3\\fscx100\\fscy100"
                            f"\\t(0,{seg_ms},\\fscx104\\fscy104)}}"
                        )
                        main_effects = (
                            f"{{\\fad(200,400)\\fscx100\\fscy100"
                            f"\\t(0,{seg_ms},\\fscx104\\fscy104)}}"
                        )

                        ass_content += (
                            f"Dialogue: {layer_shadow},{seg_start_time},{seg_end_time},Shadow,,0,0,0,,"
                            f"{shadow_effects}{karaoke_text}\n"
                        )
                        ass_content += (
                            f"Dialogue: {layer_main},{seg_start_time},{seg_end_time},Main,,0,0,0,,"
                            f"{main_effects}{karaoke_text}\n"
                        )

                    seg_start = seg_end

        return ass_content
    
    @staticmethod
    def generate_from_richsync(richsync_json: str, video_width: int = 1920, video_height: int = 1080, font_path: str = "legacy/karaoke/fonts/SpaceMono-Regular.ttf", offset_seconds: float = 0.0, progressive_fill: bool = True, primary_colour: str = "&H0000FFFF", secondary_colour: str = "&H00CCCCCC", outline_colour: str = "&HA64DFF") -> str:
        """
        Generate ASS karaoke format with word-level timing from Richsync JSON.
        Uses professional styling matching the existing karaoke generator.
        
        Args:
            richsync_json: Richsync JSON string
            video_width: Video width in pixels
            video_height: Video height in pixels
            font_path: Path to font file
            offset_seconds: Time offset to apply to all timestamps (in seconds)
            
        Returns:
            ASS format string with karaoke effects
        """
        try:
            data = json.loads(richsync_json)
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse richsync JSON: {e}")
            return ""
        
        if not isinstance(data, list):
            print("⚠️  Richsync data is not a list")
            return ""
        
        # ASS header with professional styling
        ass_content = f"""[Script Info]
Title: Karaoke with Space Mono
ScriptType: v4.00+
Collisions: Normal
PlayResX: {video_width}
PlayResY: {video_height}
Timer: 100.0000

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Main,{font_path},76,{primary_colour},{secondary_colour},{outline_colour},&H64000000,1,0,0,0,100,100,0,0,1,3,0,2,80,80,450,1
Style: Shadow,{font_path},76,&H00FFFFFF,&HFFFFFFFF,&HFFFFFF,&H64000000,1,0,0,0,100,100,0,0,1,0,0,2,80,80,450,1
Style: MainTop,{font_path},76,{primary_colour},{secondary_colour},{outline_colour},&H64000000,1,0,0,0,100,100,0,0,1,3,0,8,80,80,450,1
Style: ShadowTop,{font_path},76,&H00FFFFFF,&HFFFFFFFF,&HFFFFFF,&H64000000,1,0,0,0,100,100,0,0,1,0,0,8,80,80,450,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        
        # Generate dialogue lines with karaoke effects
        # Split long lines into chunks of max 5 words
        max_words_per_line = 6
        min_display_duration = 2.5  # Minimum seconds to display each line
        overlap_duration = 0.5  # Seconds to keep line visible after singing ends

        lane_clearance = 0.05
        karaoke_tag = "\\kf" if progressive_fill else "\\k"
        layer_chorus_shadow = 0
        layer_chorus_main = 1
        layer_shadow = 2
        layer_main = 3
        margin_l = 80
        margin_r = 80
        font_size = 76
        max_scale = 1.20
        avg_char_width = float(font_size) * 0.48 * max_scale
        available_w = max(1.0, float(video_width - margin_l - margin_r))
        max_chars_per_line = max(10, int(available_w / max(avg_char_width, 1.0)))
        chunks = []

        for entry in data:
            if not isinstance(entry, dict):
                continue
            
            line_start = entry.get('ts', 0)
            line_end = entry.get('te', line_start + 3)
            char_list = entry.get('l', [])
            
            if not char_list:
                continue
            
            # Group characters into words (split by spaces)
            words = []
            current_word = []
            
            for char_entry in char_list:
                if not isinstance(char_entry, dict):
                    continue
                
                text = char_entry.get('c', '')
                offset = char_entry.get('o', 0)
                
                if text.strip() == '':
                    # Space - end current word
                    if current_word:
                        words.append(current_word)
                        current_word = []
                else:
                    # Add character to current word
                    current_word.append({'text': text, 'offset': offset, 'entry': char_entry})
            
            # Don't forget last word
            if current_word:
                words.append(current_word)
            
            # Split words into chunks of max_words_per_line
            for chunk_idx in range(0, len(words), max_words_per_line):
                chunk_words = words[chunk_idx:chunk_idx + max_words_per_line]
                
                if not chunk_words:
                    continue
                
                # Calculate chunk timing
                chunk_start_offset = chunk_words[0][0]['offset']
                line_base = line_start + offset_seconds
                chunk_start = line_base + chunk_start_offset
                
                # Find natural end time (when singing ends)
                if chunk_idx + max_words_per_line < len(words):
                    natural_end_offset = words[chunk_idx + max_words_per_line][0]['offset']
                    natural_end = line_base + natural_end_offset
                else:
                    natural_end = line_end + offset_seconds
                
                # Calculate natural duration
                natural_duration = natural_end - chunk_start
                
                # Apply minimum display duration and overlap
                display_duration = max(natural_duration, min_display_duration)
                chunk_end = chunk_start + display_duration + overlap_duration

                last_char_offset = chunk_words[-1][-1]['offset']
                sung_end = line_base + last_char_offset

                chunks.append({
                    'line_start': line_start,
                    'line_base': line_base,
                    'chunk_start': chunk_start,
                    'chunk_end': chunk_end,
                    'sung_end': sung_end,
                    'chunk_words': chunk_words,
                })

        for i in range(len(chunks)):
            j = i + 2
            if j >= len(chunks):
                continue
            max_end = chunks[j]['chunk_start'] - lane_clearance
            max_end = max(max_end, chunks[i]['sung_end'] + lane_clearance)
            if chunks[i]['chunk_end'] > max_end:
                chunks[i]['chunk_end'] = max_end

        for line_counter, ch in enumerate(chunks):
            chunk_start = ch['chunk_start']
            chunk_end = ch['chunk_end']
            line_start = ch['line_start']
            line_base = ch['line_base']
            chunk_words = ch['chunk_words']

            if chunk_end <= chunk_start:
                continue

            chunk_duration_ms = int((chunk_end - chunk_start) * 1000)

            word_texts_all = ["".join((cd.get('text') or '') for cd in w) for w in chunk_words]

            chorus_start_offset: Optional[float] = None
            chorus_end_offset: Optional[float] = None
            chorus_text = ""
            chorus_split_idx: Optional[int] = None
            chorus_span_start: Optional[int] = None
            chorus_span_end: Optional[int] = None
            if word_texts_all:
                last_has_close = (')' in word_texts_all[-1])
                open_idx = None
                for i in range(len(word_texts_all) - 1, -1, -1):
                    if '(' in word_texts_all[i]:
                        open_idx = i
                        break
                if last_has_close and open_idx is not None and open_idx >= max(0, len(word_texts_all) - 8):
                    chorus_split_idx = int(open_idx)
                else:
                    open_any: Optional[int] = None
                    close_any: Optional[int] = None
                    for wi, wt in enumerate(word_texts_all):
                        if open_any is None and '(' in wt:
                            open_any = int(wi)
                        if open_any is not None and ')' in wt:
                            close_any = int(wi)
                    if open_any is not None and close_any is not None and close_any >= open_any:
                        chorus_words_span = chunk_words[open_any:close_any + 1]
                        if chorus_words_span:
                            chorus_span_start = int(open_any)
                            chorus_span_end = int(close_any)
                            chorus_start_offset = chorus_words_span[0][0]['offset']
                            chorus_end_offset = chorus_words_span[-1][-1]['offset']
                            chorus_text = " ".join(word_texts_all[open_any:close_any + 1])
                            chorus_text = chorus_text.replace('(', ' ').replace(')', ' ')
                            chorus_text = re.sub(r"\s+", " ", chorus_text).strip()
                            chorus_text = ASSKaraokeGenerator._cap_first_letter(chorus_text)

            main_chunk_words = chunk_words
            main_word_texts = word_texts_all
            if chorus_split_idx is not None and chorus_split_idx < len(chunk_words):
                chorus_words = chunk_words[chorus_split_idx:]
                main_chunk_words = chunk_words[:chorus_split_idx]
                main_word_texts = word_texts_all[:chorus_split_idx]
                if chorus_words:
                    chorus_start_offset = chorus_words[0][0]['offset']
                    chorus_end_offset = chorus_words[-1][-1]['offset']
                    chorus_text = " ".join(word_texts_all[chorus_split_idx:])
                    chorus_text = chorus_text.replace('(', ' ').replace(')', ' ')
                    chorus_text = re.sub(r"\s+", " ", chorus_text).strip()
                    chorus_text = ASSKaraokeGenerator._cap_first_letter(chorus_text)

            word_visible = [True for _ in range(len(main_chunk_words))]
            visible_indices = list(range(len(main_chunk_words)))
            visible_word_texts = list(main_word_texts)
            if chorus_split_idx is None and chorus_span_start is not None and chorus_span_end is not None:
                word_visible = [not (chorus_span_start <= wi <= chorus_span_end) for wi in range(len(chunk_words))]
                visible_indices = [wi for wi, vis in enumerate(word_visible) if vis]
                visible_word_texts = [word_texts_all[wi] for wi in visible_indices]

            total_non_space = sum(len(re.sub(r"\s+", "", t)) for t in visible_word_texts)
            wrap_after: Optional[int] = None
            if total_non_space > max_chars_per_line and len(main_word_texts) > 1:
                candidates = []
                running = 0
                for wi, wt in enumerate(visible_word_texts[:-1]):
                    running += len(re.sub(r"\s+", "", wt))
                    if running <= 0 or running >= total_non_space:
                        continue
                    score = max(running, total_non_space - running) * 100 + abs(running - (total_non_space - running))
                    candidates.append((score, wi))
                if candidates:
                    _, wrap_after_visible = min(candidates)
                    if 0 <= int(wrap_after_visible) < len(visible_indices):
                        wrap_after = int(visible_indices[int(wrap_after_visible)])

            karaoke_text = ""
            lead_cap_done = False
            effective_end_offset = (chunk_end - line_base)
            if chorus_split_idx is not None and chorus_start_offset is not None:
                effective_end_offset = min(float(effective_end_offset), float(chorus_start_offset))

            for word_idx, word in enumerate(main_chunk_words):
                for char_idx, char_data in enumerate(word):
                    text = char_data['text']
                    offset = char_data['offset']

                    if word_idx < len(word_visible) and not word_visible[word_idx]:
                        text = ""

                    if chorus_text and text in ("(", ")"):
                        text = ""

                    if not lead_cap_done and text and any(ch.isalpha() for ch in text):
                        text = ASSKaraokeGenerator._cap_first_letter(text)
                        lead_cap_done = True

                    if char_idx + 1 < len(word):
                        next_offset = word[char_idx + 1]['offset']
                    elif word_idx + 1 < len(main_chunk_words):
                        next_offset = main_chunk_words[word_idx + 1][0]['offset']
                    else:
                        next_offset = effective_end_offset

                    duration_cs = max(int(max(0.01, next_offset - offset) * 100), 1)
                    karaoke_text += "{" + karaoke_tag + str(duration_cs) + "}" + text

                if word_idx < len(main_chunk_words) - 1:
                    cur_vis = (word_idx < len(word_visible) and word_visible[word_idx])
                    if cur_vis and any(word_visible[j] for j in range(word_idx + 1, len(word_visible))):
                        if wrap_after is not None and word_idx == wrap_after:
                            karaoke_text += "\\N"
                        else:
                            karaoke_text += " "

            start_time = ASSKaraokeGenerator.format_ass_timestamp(chunk_start)
            end_time = ASSKaraokeGenerator.format_ass_timestamp(chunk_end)

            is_top = (line_counter % 2 == 1)
            shadow_style = "ShadowTop" if is_top else "Shadow"
            main_style = "MainTop" if is_top else "Main"

            if chorus_text and chorus_start_offset is not None and chorus_end_offset is not None:
                chorus_start = line_base + float(chorus_start_offset)
                chorus_end = min(chunk_end, (line_base + float(chorus_end_offset) + 0.6))
                if chorus_end > chorus_start:
                    chorus_words = re.findall(r"\S+", ASSKaraokeGenerator._cap_first_letter(chorus_text))
                    if chorus_words:
                        backing_pre_roll_seconds = 0.25
                        backing_post_roll_seconds = 0.90
                        base_y = int(video_height * 0.50)
                        fall_px = 160

                        total = max(0.2, float(chorus_end - chorus_start))
                        stagger = min(0.35, (total / max(len(chorus_words), 1)) * 0.9)
                        hold_extra = 2.0
                        chorus_hold_end = min(chunk_end, chorus_end + hold_extra)

                        font_size_chorus = 190
                        avg_char_w = float(font_size_chorus) * 0.48
                        space_w = avg_char_w * 0.9
                        word_ws = [max(1.0, len(w) * avg_char_w) for w in chorus_words]
                        total_w = sum(word_ws) + space_w * max(len(chorus_words) - 1, 0)
                        available_w = float(video_width) * 0.92
                        scale = min(1.0, available_w / max(total_w, 1.0))
                        scale_pct = int(round(scale * 100.0))
                        space_w_s = space_w * scale
                        word_ws_s = [ww * scale for ww in word_ws]
                        total_w_s = total_w * scale
                        start_x = (float(video_width) / 2.0) - (total_w_s / 2.0)

                        for wi, w in enumerate(chorus_words):
                            ws = chorus_start + (wi * stagger)
                            we = chorus_hold_end
                            if we <= ws:
                                continue

                            ws_disp = max(chunk_start, ws - backing_pre_roll_seconds)
                            upper_end = chunk_end + backing_post_roll_seconds
                            if line_counter + 2 < len(chunks):
                                upper_end = min(
                                    upper_end,
                                    chunks[line_counter + 2]['chunk_start'] - lane_clearance,
                                )
                            we_disp = min(upper_end, we + backing_post_roll_seconds)
                            if we_disp <= ws_disp:
                                continue

                            dur_ms = int((we_disp - ws_disp) * 1000)
                            appear_ms = 220
                            fall_dur_ms = min(1200, max(600, int(dur_ms * 0.45)))
                            fall_t1 = max(appear_ms, dur_ms - fall_dur_ms)
                            fall_t2 = dur_ms
                            fade_out_t1 = fall_t1
                            fall_mid = int(fall_t1 + ((fall_t2 - fall_t1) * 0.5))

                            x_center = start_x + (sum(word_ws_s[:wi]) + (space_w_s * wi) + (word_ws_s[wi] / 2.0))
                            x = int(x_center)
                            y0 = base_y
                            y1 = base_y + fall_px
                            y_mid = int(y0 + ((y1 - y0) * 0.65))

                            ws_t = ASSKaraokeGenerator.format_ass_timestamp(ws_disp)
                            we_t = ASSKaraokeGenerator.format_ass_timestamp(we_disp)

                            chorus_shadow_fx = (
                                f"{{\\an5\\bord0\\shad0"
                                f"\\alpha&HFF&\\blur5\\fs190\\fscx{scale_pct}\\fscy{scale_pct}"
                                f"\\1c&H000000&"
                                f"\\t(0,{appear_ms},0.7,\\alpha&H44&)"
                                f"\\pos({x},{y0})"
                                f"\\t({fall_t1},{fall_mid},1.6,\\pos({x},{y_mid}))"
                                f"\\t({fall_mid},{fall_t2},0.6,\\pos({x},{y1}))"
                                f"\\3c&H000000&\\4c&H000000&"
                                f"\\t({fade_out_t1},{fall_t2},1.4,\\alpha&HFF&\\blur5)}}"
                            )
                            chorus_main_fx = (
                                f"{{\\an5\\bord0\\shad0"
                                f"\\alpha&HFF&\\blur4\\fs190\\fscx{scale_pct}\\fscy{scale_pct}"
                                f"\\1c{outline_colour}&"
                                f"\\t(0,{appear_ms},0.7,\\alpha&H44&)"
                                f"\\pos({x},{y0})"
                                f"\\t({fall_t1},{fall_mid},1.6,\\pos({x},{y_mid}))"
                                f"\\t({fall_mid},{fall_t2},0.6,\\pos({x},{y1}))"
                                f"\\t({fade_out_t1},{fall_t2},1.4,\\alpha&HFF&\\blur4)}}"
                            )
                            ass_content += f"Dialogue: {layer_chorus_shadow},{ws_t},{we_t},{shadow_style},,0,0,0,,{chorus_shadow_fx}{w}\n"
                            ass_content += f"Dialogue: {layer_chorus_main},{ws_t},{we_t},{main_style},,0,0,0,,{chorus_main_fx}{w}\n"

            if karaoke_text:
                shadow_effects = f"{{\\fad(500,500)\\blur10\\3c&H000000&\\4c&H000000&\\fscx100\\fscy100\\t(0,{chunk_duration_ms},\\fscx120\\fscy120)}}"
                ass_content += f"Dialogue: {layer_shadow},{start_time},{end_time},{shadow_style},,0,0,0,,{shadow_effects}{karaoke_text}\n"

                main_effects = f"{{\\fad(500,500)\\fscx100\\fscy100\\t(0,{chunk_duration_ms},\\fscx120\\fscy120)}}"
                ass_content += f"Dialogue: {layer_main},{start_time},{end_time},{main_style},,0,0,0,,{main_effects}{karaoke_text}\n"

        return ass_content
    
    @staticmethod
    def generate_from_richsync_reel(
        richsync_json: str,
        video_width: int = 1080,
        video_height: int = 1920,
        font_path: str = "legacy/karaoke/fonts/SpaceMono-Regular.ttf",
        offset_seconds: float = 0.0,
        region_x: int = 70,
        region_y: int = 520,
        region_w: int = 946,
        region_h: int = 320,
        progressive_fill: bool = True,
        primary_colour: str = "&H0000FFFF",
        secondary_colour: str = "&H00CCCCCC",
        outline_colour: str = "&HA64DFF",
    ) -> str:
        """Generate ASS for Instagram reel layout.

        Differences from generate_from_richsync:
        - Single line at a time, always in the same region.
        - Simple fade-in only (no fade-out).
        - Text constrained to a rectangle starting at (region_x, region_y)
          with size (region_w, region_h).
        """
        try:
            data = json.loads(richsync_json)
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse richsync JSON (reel): {e}")
            return ""
        
        if not isinstance(data, list):
            print("⚠️  Richsync data is not a list (reel)")
            return ""

        # Compute margins for the reel text region
        margin_l = region_x
        margin_r = max(0, video_width - (region_x + region_w))
        # Top-centered alignment with MarginV slightly above box center for
        # a visually pleasing vertical position.
        margin_v = region_y + region_h // 2 - 40

        ass_content = f"""[Script Info]
Title: Karaoke Reel with Space Mono
ScriptType: v4.00+
Collisions: Normal
PlayResX: {video_width}
PlayResY: {video_height}
Timer: 100.0000

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Main,{font_path},76,{primary_colour},{secondary_colour},{outline_colour},&H64000000,1,0,0,0,100,100,0,0,1,3,0,8,{margin_l},{margin_r},{margin_v},1
Style: Shadow,{font_path},76,&H00FFFFFF,&HFFFFFFFF,&HFFFFFF,&H64000000,1,0,0,0,100,100,0,0,1,0,0,8,{margin_l},{margin_r},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

        max_words_per_line = 5
        karaoke_tag = "\\kf" if progressive_fill else "\\k"
        layer_chorus_shadow = 0
        layer_chorus_main = 1
        layer_shadow = 2
        layer_main = 3
        font_size = 76
        max_scale = 1.04
        avg_char_width = float(font_size) * 0.48 * max_scale
        available_w = max(1.0, float(region_w))
        max_chars_per_line = max(10, int(available_w / max(avg_char_width, 1.0)))

        for entry in data:
            if not isinstance(entry, dict):
                continue

            line_start = entry.get("ts", 0)
            line_end = entry.get("te", line_start + 3)
            line_base = line_start + offset_seconds
            char_list = entry.get("l", [])

            if not char_list:
                continue

            # Group characters into words (split by spaces)
            words = []
            current_word = []

            for char_entry in char_list:
                if not isinstance(char_entry, dict):
                    continue

                text = char_entry.get("c", "")
                offset = char_entry.get("o", 0)

                if text.strip() == "":
                    if current_word:
                        words.append(current_word)
                        current_word = []
                else:
                    current_word.append({"text": text, "offset": offset})

            if current_word:
                words.append(current_word)

            # Split words into chunks (one subtitle line per chunk, sequential, no manual \N)
            for chunk_idx in range(0, len(words), max_words_per_line):
                chunk_words = words[chunk_idx:chunk_idx + max_words_per_line]
                if not chunk_words:
                    continue

                # Start time for this chunk
                chunk_start_offset = chunk_words[0][0]["offset"]
                chunk_start = line_start + chunk_start_offset + offset_seconds

                # Natural end (next chunk start or line end)
                if chunk_idx + max_words_per_line < len(words):
                    natural_end_offset = words[chunk_idx + max_words_per_line][0]["offset"]
                    natural_end = line_start + natural_end_offset + offset_seconds
                else:
                    natural_end = line_end + offset_seconds

                # Actual sung end: last character of the chunk
                last_char_offset = chunk_words[-1][-1]["offset"]
                sung_end = line_start + last_char_offset + offset_seconds

                # Duration du dernier mot (pour éventuellement garder la ligne
                # un peu plus longtemps s'il est très tenu).
                last_word_start_offset = chunk_words[-1][0]["offset"]
                last_word_duration = max(0.0, last_char_offset - last_word_start_offset)
                is_long_last_word = last_word_duration >= 0.7

                # Logique simple : on garde chaque ligne un peu après la fin
                # du dernier mot, avec un padding plus grand pour les mots
                # tenus. Pas de bridging sophistiqué entre chunks, pour éviter
                # l'empilement.
                base_padding = 0.8
                extra_padding = 0.5 if is_long_last_word else 0.0
                padding = base_padding + extra_padding
                chunk_end = min(natural_end, sung_end + padding)

                if chunk_end <= chunk_start:
                    continue

                # Duration of this chunk in milliseconds (used for transforms)
                chunk_duration_ms = int((chunk_end - chunk_start) * 1000)

                # Build karaoke text (with stable wrapping) and optional visual chorus when parentheses are a suffix
                word_texts_all = ["".join((cd.get('text') or '') for cd in w) for w in chunk_words]

                chorus_start_offset: Optional[float] = None
                chorus_end_offset: Optional[float] = None
                chorus_text = ""
                chorus_split_idx: Optional[int] = None
                chorus_span_start: Optional[int] = None
                chorus_span_end: Optional[int] = None
                if word_texts_all:
                    last_has_close = (')' in word_texts_all[-1])
                    open_idx = None
                    for i in range(len(word_texts_all) - 1, -1, -1):
                        if '(' in word_texts_all[i]:
                            open_idx = i
                            break
                    if last_has_close and open_idx is not None and open_idx >= max(0, len(word_texts_all) - 8):
                        chorus_split_idx = int(open_idx)
                    else:
                        open_any: Optional[int] = None
                        close_any: Optional[int] = None
                        for wi, wt in enumerate(word_texts_all):
                            if open_any is None and '(' in wt:
                                open_any = int(wi)
                            if open_any is not None and ')' in wt:
                                close_any = int(wi)
                        if open_any is not None and close_any is not None and close_any >= open_any:
                            chorus_words_span = chunk_words[open_any:close_any + 1]
                            if chorus_words_span:
                                chorus_span_start = int(open_any)
                                chorus_span_end = int(close_any)
                                chorus_start_offset = chorus_words_span[0][0]["offset"]
                                chorus_end_offset = chorus_words_span[-1][-1]["offset"]
                                chorus_text = " ".join(word_texts_all[open_any:close_any + 1])
                                chorus_text = chorus_text.replace('(', ' ').replace(')', ' ')
                                chorus_text = re.sub(r"\s+", " ", chorus_text).strip()
                                chorus_text = ASSKaraokeGenerator._cap_first_letter(chorus_text)

                main_chunk_words = chunk_words
                main_word_texts = word_texts_all
                if chorus_split_idx is not None and chorus_split_idx < len(chunk_words):
                    chorus_words = chunk_words[chorus_split_idx:]
                    main_chunk_words = chunk_words[:chorus_split_idx]
                    main_word_texts = word_texts_all[:chorus_split_idx]
                    if chorus_words:
                        chorus_start_offset = chorus_words[0][0]["offset"]
                        chorus_end_offset = chorus_words[-1][-1]["offset"]
                        chorus_text = " ".join(word_texts_all[chorus_split_idx:])
                        chorus_text = chorus_text.replace('(', ' ').replace(')', ' ')
                        chorus_text = re.sub(r"\s+", " ", chorus_text).strip()
                        chorus_text = ASSKaraokeGenerator._cap_first_letter(chorus_text)

                word_visible = [True for _ in range(len(main_chunk_words))]
                visible_indices = list(range(len(main_chunk_words)))
                visible_word_texts = list(main_word_texts)
                if chorus_split_idx is None and chorus_span_start is not None and chorus_span_end is not None:
                    word_visible = [not (chorus_span_start <= wi <= chorus_span_end) for wi in range(len(chunk_words))]
                    visible_indices = [wi for wi, vis in enumerate(word_visible) if vis]
                    visible_word_texts = [word_texts_all[wi] for wi in visible_indices]

                total_non_space = sum(len(re.sub(r"\s+", "", t)) for t in visible_word_texts)
                wrap_after: Optional[int] = None
                if total_non_space > max_chars_per_line and len(main_word_texts) > 1:
                    candidates = []
                    running = 0
                    for wi, wt in enumerate(visible_word_texts[:-1]):
                        running += len(re.sub(r"\s+", "", wt))
                        if running <= 0 or running >= total_non_space:
                            continue
                        score = max(running, total_non_space - running) * 100 + abs(running - (total_non_space - running))
                        candidates.append((score, wi))
                    if candidates:
                        _, wrap_after_visible = min(candidates)
                        if 0 <= int(wrap_after_visible) < len(visible_indices):
                            wrap_after = int(visible_indices[int(wrap_after_visible)])

                karaoke_text = ""
                lead_cap_done = False
                effective_end_offset = (chunk_end - line_base)
                if chorus_split_idx is not None and chorus_start_offset is not None:
                    effective_end_offset = min(float(effective_end_offset), float(chorus_start_offset))

                for word_idx, word in enumerate(main_chunk_words):
                    for char_idx, char_data in enumerate(word):
                        text = char_data["text"]
                        offset = char_data["offset"]

                        if word_idx < len(word_visible) and not word_visible[word_idx]:
                            text = ""

                        if chorus_text and text in ("(", ")"):
                            text = ""

                        if not lead_cap_done and text and any(ch.isalpha() for ch in text):
                            text = ASSKaraokeGenerator._cap_first_letter(text)
                            lead_cap_done = True

                        if char_idx + 1 < len(word):
                            next_offset = word[char_idx + 1]["offset"]
                        elif word_idx + 1 < len(main_chunk_words):
                            next_offset = main_chunk_words[word_idx + 1][0]["offset"]
                        else:
                            # Fallback: stretch to adjusted end time
                            next_offset = effective_end_offset

                        duration_cs = int(max(0.01, next_offset - offset) * 100)
                        karaoke_text += "{" + karaoke_tag + str(duration_cs) + "}" + text

                    if word_idx < len(main_chunk_words) - 1:
                        cur_vis = (word_idx < len(word_visible) and word_visible[word_idx])
                        if cur_vis and any(word_visible[j] for j in range(word_idx + 1, len(word_visible))):
                            if wrap_after is not None and word_idx == wrap_after:
                                karaoke_text += "\\N"
                            else:
                                karaoke_text += " "

                if chorus_text and chorus_start_offset is not None and chorus_end_offset is not None:
                    chorus_start = line_start + float(chorus_start_offset) + offset_seconds
                    chorus_end = min(chunk_end, (line_start + float(chorus_end_offset) + offset_seconds + 0.6))
                    if chorus_end > chorus_start:
                        chorus_words = re.findall(r"\S+", ASSKaraokeGenerator._cap_first_letter(chorus_text))
                        if chorus_words:
                            backing_pre_roll_seconds = 0.25
                            backing_post_roll_seconds = 0.90
                            base_y = int(region_y + (region_h * 0.50))
                            fall_px = 140

                            total = max(0.2, float(chorus_end - chorus_start))
                            stagger = min(0.32, (total / max(len(chorus_words), 1)) * 0.9)
                            hold_extra = 1.8
                            chorus_hold_end = min(chunk_end, chorus_end + hold_extra)

                            font_size_chorus = 180
                            avg_char_w = float(font_size_chorus) * 0.48
                            space_w = avg_char_w * 0.9
                            word_ws = [max(1.0, len(w) * avg_char_w) for w in chorus_words]
                            total_w = sum(word_ws) + space_w * max(len(chorus_words) - 1, 0)
                            available_w = float(video_width) * 0.92
                            scale = min(1.0, available_w / max(total_w, 1.0))
                            scale_pct = int(round(scale * 100.0))
                            space_w_s = space_w * scale
                            word_ws_s = [ww * scale for ww in word_ws]
                            total_w_s = total_w * scale
                            start_x = (float(video_width) / 2.0) - (total_w_s / 2.0)

                            for wi, w in enumerate(chorus_words):
                                ws = chorus_start + (wi * stagger)
                                we = chorus_hold_end
                                if we <= ws:
                                    continue

                                ws_disp = max(chunk_start, ws - backing_pre_roll_seconds)
                                upper_end = min(chunk_end + backing_post_roll_seconds, natural_end - 0.05)
                                upper_end = max(chunk_end, upper_end)
                                we_disp = min(upper_end, we + backing_post_roll_seconds)
                                if we_disp <= ws_disp:
                                    continue

                                dur_ms = int((we_disp - ws_disp) * 1000)
                                appear_ms = 200
                                fall_dur_ms = min(1100, max(550, int(dur_ms * 0.45)))
                                fall_t1 = max(appear_ms, dur_ms - fall_dur_ms)
                                fall_t2 = dur_ms
                                fade_out_t1 = fall_t1
                                fall_mid = int(fall_t1 + ((fall_t2 - fall_t1) * 0.5))

                                x_center = start_x + (sum(word_ws_s[:wi]) + (space_w_s * wi) + (word_ws_s[wi] / 2.0))
                                x = int(x_center)
                                y0 = base_y
                                y1 = base_y + fall_px
                                y_mid = int(y0 + ((y1 - y0) * 0.65))

                                ws_t = ASSKaraokeGenerator.format_ass_timestamp(ws_disp)
                                we_t = ASSKaraokeGenerator.format_ass_timestamp(we_disp)

                                chorus_shadow_fx = (
                                    f"{{\\an5\\bord0\\shad0"
                                    f"\\alpha&HFF&\\blur5\\fs180\\fscx{scale_pct}\\fscy{scale_pct}"
                                    f"\\1c&H000000&"
                                    f"\\t(0,{appear_ms},0.7,\\alpha&H44&)"
                                    f"\\pos({x},{y0})"
                                    f"\\t({fall_t1},{fall_mid},1.6,\\pos({x},{y_mid}))"
                                    f"\\t({fall_mid},{fall_t2},0.6,\\pos({x},{y1}))"
                                    f"\\3c&H000000&\\4c&H000000&"
                                    f"\\t({fade_out_t1},{fall_t2},1.4,\\alpha&HFF&\\blur5)}}"
                                )
                                chorus_main_fx = (
                                    f"{{\\an5\\bord0\\shad0"
                                    f"\\alpha&HFF&\\blur4\\fs180\\fscx{scale_pct}\\fscy{scale_pct}"
                                    f"\\1c{outline_colour}&"
                                    f"\\t(0,{appear_ms},0.7,\\alpha&H44&)"
                                    f"\\pos({x},{y0})"
                                    f"\\t({fall_t1},{fall_mid},1.6,\\pos({x},{y_mid}))"
                                    f"\\t({fall_mid},{fall_t2},0.6,\\pos({x},{y1}))"
                                    f"\\t({fade_out_t1},{fall_t2},1.4,\\alpha&HFF&\\blur4)}}"
                                )
                                ass_content += (
                                    f"Dialogue: {layer_chorus_shadow},{ws_t},{we_t},Shadow,,0,0,0,,"
                                    f"{chorus_shadow_fx}{w}\n"
                                )
                                ass_content += (
                                    f"Dialogue: {layer_chorus_main},{ws_t},{we_t},Main,,0,0,0,,"
                                    f"{chorus_main_fx}{w}\n"
                                )

                # Effet: léger zoom continu sur toute la durée du chunk,
                # inspiré du mode generate mais plus subtil. On part de 100%
                # et on monte à ~104% sur chunk_duration_ms.
                shadow_effects = (
                    f"{{\\fad(200,400)\\blur3\\fscx100\\fscy100"
                    f"\\t(0,{chunk_duration_ms},\\fscx104\\fscy104)}}"
                )
                main_effects = (
                    f"{{\\fad(200,400)\\fscx100\\fscy100"
                    f"\\t(0,{chunk_duration_ms},\\fscx104\\fscy104)}}"
                )

                start_time = ASSKaraokeGenerator.format_ass_timestamp(chunk_start)
                end_time = ASSKaraokeGenerator.format_ass_timestamp(chunk_end)
                if karaoke_text:
                    ass_content += (
                        f"Dialogue: {layer_shadow},{start_time},{end_time},Shadow,,0,0,0,,"
                        f"{shadow_effects}{karaoke_text}\n"
                    )
                    ass_content += (
                        f"Dialogue: {layer_main},{start_time},{end_time},Main,,0,0,0,,"
                        f"{main_effects}{karaoke_text}\n"
                    )

        return ass_content
    
    @staticmethod
    def format_ass_timestamp(seconds: float) -> str:
        """
        Format seconds to ASS timestamp (H:MM:SS.cc).
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centiseconds = int((seconds % 1) * 100)
        
        return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"


class SRTGenerator:
    """Generate SRT subtitle format from synced lines."""
    
    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """
        Format seconds to SRT timestamp (HH:MM:SS,mmm).
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp string
        """
        td = timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        secs = int(td.total_seconds() % 60)
        millis = int((td.total_seconds() % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    @staticmethod
    def generate(synced_lines: List[SyncedLine]) -> str:
        """
        Generate SRT format from synced lines.
        
        Args:
            synced_lines: List of SyncedLine objects
            
        Returns:
            SRT format string
        """
        srt_lines = []
        
        for i, line in enumerate(synced_lines, 1):
            start_ts = SRTGenerator.format_timestamp(line.start)
            end_ts = SRTGenerator.format_timestamp(line.end)
            
            srt_lines.append(f"{i}")
            srt_lines.append(f"{start_ts} --> {end_ts}")
            srt_lines.append(line.text)
            srt_lines.append("")  # Empty line between entries
        
        return '\n'.join(srt_lines)


def get_synced_lyrics(
    track_name: str,
    artist_name: str,
    api_key: str,
    album_name: Optional[str] = None,
    prefer_richsync: bool = False,
    output_format: str = 'srt',
    target_duration: Optional[int] = None,
    pick: Optional[int] = None,
    musixmatch_track_id: Optional[int] = None
) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Get synced lyrics from Musixmatch and convert to desired format.
    
    Args:
        track_name: Song title
        artist_name: Artist name
        api_key: Musixmatch API key
        album_name: Album name (optional, helps with matching)
        prefer_richsync: Prefer richsync over subtitle if available
        output_format: Output format ('srt', 'lrc', 'json')
        
    Returns:
        Tuple of (synced_lyrics_content, metadata_dict)
    """
    client = MusixmatchClient(api_key)
    
    # Search for track (2-pass: with album, then without album if too few results)
    tracks = client.search_track(track_name, artist_name, album_name)
    if album_name and len(tracks) <= 1:
        more = client.search_track(track_name, artist_name, None)
        if more:
            existing = {int(t.track_id) for t in tracks}
            tracks.extend([t for t in more if int(t.track_id) not in existing])
    
    if not tracks:
        print("❌ No tracks found")
        return None, None

    # Fallback: matcher.track.get can return a better canonical track (duration, non-live)
    matcher_track = client.matcher_track_get(track_name, artist_name, None)
    if matcher_track:
        existing = {int(t.track_id) for t in tracks}
        if int(matcher_track.track_id) not in existing:
            tracks.insert(0, matcher_track)
    
    # Display results (hydrate missing durations for readability)
    client.hydrate_track_lengths(tracks, limit=5)
    print(f"\n📋 Search results:")
    for i, track in enumerate(tracks, 1):
        sync_info = []
        if track.has_subtitles:
            sync_info.append("subtitles")
        if track.has_richsync:
            sync_info.append("richsync")
        if track.has_lyrics:
            sync_info.append("lyrics")
        
        sync_str = ", ".join(sync_info) if sync_info else "no sync data"
        dur_str = f"{track.track_length}s" if track.track_length and int(track.track_length) > 0 else "?"
        print(f"  {i}. {track.artist_name} - {track.track_name}")
        print(f"     Album: {track.album_name}, Duration: {dur_str}")
        print(f"     Available: {sync_str}")
    
    # If we need to match by duration, ensure track lengths are hydrated
    if target_duration:
        client.hydrate_track_lengths(tracks)

        if all((not t.track_length or int(t.track_length) <= 0) for t in tracks):
            matcher_track = client.matcher_track_get(track_name, artist_name, None)
            if matcher_track and (matcher_track.track_length and int(matcher_track.track_length) > 0):
                existing = {int(t.track_id) for t in tracks}
                if int(matcher_track.track_id) not in existing:
                    tracks.insert(0, matcher_track)

    best_match = None

    if pick is not None:
        idx = int(pick) - 1
        if idx < 0 or idx >= len(tracks):
            print(f"❌ Invalid --pick value: {pick}. Must be between 1 and {len(tracks)}")
            return None, None
        best_match = tracks[idx]

    if best_match is None and musixmatch_track_id is not None:
        for t in tracks:
            if int(t.track_id) == int(musixmatch_track_id):
                best_match = t
                break
        if best_match is None:
            print(f"❌ Musixmatch track id {musixmatch_track_id} not found in search results")
            return None, None

    def _norm(s: str) -> str:
        s = s.lower()
        s = re.sub(r"\(.*?\)", " ", s)
        s = re.sub(r"[^a-z0-9]+", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    def _score(t: MusixmatchTrack) -> Tuple[int, int, int]:
        name = t.track_name or ''
        name_n = _norm(name)
        q_n = _norm(track_name)
        bad = 0
        if 'live' in name_n:
            bad += 2
        if 'remaster' in name_n or 'remastered' in name_n:
            bad += 1
        if 'demo' in name_n:
            bad += 1
        exact = 1 if name_n == q_n else 0
        prefix = 1 if name_n.startswith(q_n) else 0
        has_sync = 1 if (t.has_richsync or t.has_subtitles) else 0
        return (bad, -has_sync, -(exact + prefix))

    if best_match is None and target_duration:
        print(f"\n🎯 Target duration: {target_duration}s - picking closest match with synced lyrics")
        synced_tracks = [t for t in tracks if (t.has_subtitles or t.has_richsync)]
        known_duration_tracks = [t for t in synced_tracks if (t.track_length and int(t.track_length) > 0)]
        if synced_tracks:
            if known_duration_tracks:
                best_match = min(known_duration_tracks, key=lambda t: abs(t.track_length - target_duration))
                print(f"   Best match: {best_match.track_name} ({best_match.track_length}s, diff: {abs(best_match.track_length - target_duration)}s)")
            else:
                best_match = sorted(synced_tracks, key=_score)[0]
                print(f"   Best match (no durations available): {best_match.track_name}")
        else:
            best_match = tracks[0]

    if best_match is None:
        best_match = tracks[0]
    
    print(f"\n✅ Using: {best_match.artist_name} - {best_match.track_name}")
    
    # Try to get synced lyrics
    synced_lines = None
    
    desired_len = best_match.track_length if (best_match.track_length and int(best_match.track_length) > 0) else target_duration

    if prefer_richsync and best_match.has_richsync:
        richsync_data = client.get_richsync(best_match.track_id, desired_len)
        if richsync_data:
            output = richsync_data
            metadata = {
                'track_name': best_match.track_name,
                'artist_name': best_match.artist_name,
                'album_name': best_match.album_name,
                'track_id': best_match.track_id,
                'duration': best_match.track_length,
                'line_count': 0,
                'source': 'musixmatch',
                'format': 'richsync',
                'richsync': True
            }
            return output, metadata
    
    if best_match.has_subtitles:
        lrc_content = client.get_subtitle(best_match.track_id, 'lrc', desired_len)
        if lrc_content:
            synced_lines = LRCParser.parse(lrc_content)
        
        # If LRC failed, try JSON format
        if not synced_lines:
            subtitle_data = client.get_subtitle(best_match.track_id, 'mxm', desired_len)
            if subtitle_data:
                # MXM format is JSON, parse it
                try:
                    mxm_data = json.loads(subtitle_data)
                    # Convert to synced lines (format may vary)
                    synced_lines = []  # TODO: Parse MXM format if needed
                except:
                    pass
    
    if not synced_lines:
        print("❌ No synced lyrics available for this track")
        return None, None
    
    print(f"\n✅ Retrieved {len(synced_lines)} synced lines")
    
    # Generate output
    if output_format == 'srt':
        output = SRTGenerator.generate(synced_lines)
    elif output_format == 'lrc':
        # Convert back to LRC
        lrc_lines = []
        for line in synced_lines:
            minutes = int(line.start // 60)
            seconds = line.start % 60
            lrc_lines.append(f"[{minutes:02d}:{seconds:05.2f}]{line.text}")
        output = '\n'.join(lrc_lines)
    elif output_format == 'json':
        output = json.dumps([
            {'start': line.start, 'end': line.end, 'text': line.text}
            for line in synced_lines
        ], indent=2)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    
    # Metadata
    metadata = {
        'track_name': best_match.track_name,
        'artist_name': best_match.artist_name,
        'album_name': best_match.album_name,
        'track_id': best_match.track_id,
        'duration': best_match.track_length,
        'line_count': len(synced_lines),
        'source': 'musixmatch'
    }
    
    return output, metadata


def get_track_info_from_airtable(record_id: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[int]]:
    """
    Get track information from Airtable record.
    
    Args:
        record_id: Airtable record ID
        
    Returns:
        Tuple of (track_name, artist_name, album_name)
    """
    try:
        from pyairtable import Api
        
        # Get Airtable credentials from environment
        api_key = os.getenv('AIRTABLE_API_KEY')
        base_id = os.getenv('AIRTABLE_BASE_ID')
        table_name = os.getenv('AIRTABLE_TABLE_NAME', 'Tracks')
        
        if not api_key or not base_id:
            raise Exception("AIRTABLE_API_KEY and AIRTABLE_BASE_ID must be set in .env")
        
        print(f"📋 Fetching record {record_id} from Airtable...")
        api = Api(api_key)
        table = api.table(base_id, table_name)
        record = table.get(record_id)
        
        fields = record.get('fields', {})
        
        # Extract track info - try multiple field name variations
        track_name = (
            fields.get('Title') or
            fields.get('Track') or 
            fields.get('Track name') or
            fields.get('Song')
        )
        
        artist_name = (
            fields.get('Name (from Artist)') or
            fields.get('Artist') or
            fields.get('Artist name') or
            fields.get('Artists')
        )
        
        album_name = (
            fields.get('Album') or
            fields.get('Album name') or
            fields.get('Spotify release name')
        )
        
        duration = (
            fields.get('Duration (GDrive)') or
            fields.get('Duration (Spotify)') or
            fields.get('Duration (from GDrive Audio files)') or
            fields.get('Duration') or
            fields.get('Track duration')
        )

        # Handle list fields (from lookups)
        if isinstance(track_name, list):
            track_name = track_name[0] if track_name else None
        if isinstance(artist_name, list):
            artist_name = artist_name[0] if artist_name else None
        if isinstance(album_name, list):
            album_name = album_name[0] if album_name else None
        if isinstance(duration, list):
            duration = duration[0] if duration else None

        if duration and isinstance(duration, str):
            try:
                duration = int(float(duration))
            except (ValueError, TypeError):
                duration = None
        elif duration and isinstance(duration, (int, float)):
            duration = int(duration)
        
        if track_name and artist_name:
            print(f"✅ Found: {artist_name} - {track_name}")
            if album_name:
                print(f"   Album: {album_name}")
            if duration:
                print(f"   🎵 Duration: {duration}s")
            return track_name, artist_name, album_name, duration
        else:
            print(f"⚠️  Missing track or artist info in record")
            print(f"   Track: {track_name}")
            print(f"   Artist: {artist_name}")
            return None, None, None, None
            
    except Exception as e:
        print(f"❌ Error fetching from Airtable: {e}")
        return None, None, None, None


def save_to_airtable(record_id: str, srt_content: str, metadata: Dict) -> bool:
    """
    Save synced lyrics to Airtable record.
    
    Args:
        record_id: Airtable record ID
        srt_content: SRT content to save
        metadata: Metadata dict
        
    Returns:
        True if successful
    """
    try:
        from pyairtable import Api
        
        # Get Airtable credentials from environment
        api_key = os.getenv('AIRTABLE_API_KEY')
        base_id = os.getenv('AIRTABLE_BASE_ID')
        table_name = os.getenv('AIRTABLE_TABLE_NAME', 'Tracks')
        
        if not api_key or not base_id:
            raise Exception("AIRTABLE_API_KEY and AIRTABLE_BASE_ID must be set in .env")
        
        print(f"\n💾 Saving to Airtable record {record_id}...")
        api = Api(api_key)
        table = api.table(base_id, table_name)
        
        # Extract plain text lyrics from SRT for the lyrics field
        import re
        # Remove SRT formatting (numbers, timestamps, empty lines)
        lyrics_text = re.sub(r'^\d+\s*$', '', srt_content, flags=re.MULTILINE)
        lyrics_text = re.sub(r'^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\s*$', '', lyrics_text, flags=re.MULTILINE)
        lyrics_text = re.sub(r'\n{3,}', '\n\n', lyrics_text)
        lyrics_text = lyrics_text.strip()
        
        update_fields = {
            'SRT (Musixmatch)': srt_content,
            'Musixmatch lyrics': lyrics_text
        }

        track_id = metadata.get('track_id') if isinstance(metadata, dict) else None
        if track_id is not None:
            update_fields['Musixmatch Track ID'] = int(track_id)

        update_fields['Musixmatch Fetch Date'] = datetime.now().isoformat()

        result = table.update(record_id, update_fields)
        
        if result:
            print("✅ Saved to Airtable successfully")
            return True
        else:
            print("⚠️  Could not save to Airtable")
            return False
            
    except Exception as e:
        print(f"❌ Error saving to Airtable: {e}")
        return False


def main():
    """Example usage."""
    import argparse
    from dotenv import load_dotenv
    
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='Get synced lyrics from Musixmatch')
    
    # Input mode: either manual or from Airtable
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--track', help='Track name (use with --artist)')
    input_group.add_argument('--airtable-record', help='Airtable record ID')
    
    parser.add_argument('--artist', help='Artist name (required with --track)')
    parser.add_argument('--album', help='Album name (optional)')
    parser.add_argument('--output', help='Output file path (default: auto-generated)')
    parser.add_argument('--format', choices=['srt', 'lrc', 'json'], default='srt', help='Output format')
    parser.add_argument('--richsync', action='store_true', help='Prefer richsync over subtitle')
    parser.add_argument('--api-key', help='Musixmatch API key (or set MUSIXMATCH_API_KEY env var)')
    parser.add_argument('--save-to-airtable', action='store_true', help='Save result back to Airtable')
    parser.add_argument('--pick', type=int, default=None, help='Pick Nth Musixmatch search result (1-based)')
    parser.add_argument('--musixmatch-track-id', type=int, default=None, help='Select a Musixmatch track id from the search results')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv('MUSIXMATCH_API_KEY')
    if not api_key:
        print("❌ Error: Musixmatch API key not provided")
        print("   Set MUSIXMATCH_API_KEY environment variable or use --api-key")
        return
    
    # Determine input mode
    if args.airtable_record:
        # Fetch from Airtable
        track_name, artist_name, album_name, duration = get_track_info_from_airtable(args.airtable_record)
        if not track_name or not artist_name:
            print("❌ Could not get track info from Airtable")
            return
        
        # Auto-generate output filename if not provided
        if not args.output:
            safe_name = f"{artist_name}_{track_name}".replace(' ', '_').replace('/', '_')
            args.output = f"output/{safe_name}.{args.format}"
    else:
        # Manual input
        if not args.artist:
            print("❌ Error: --artist is required when using --track")
            return
        track_name = args.track
        artist_name = args.artist
        album_name = args.album
        
        if not args.output:
            args.output = 'output.srt'
    
    # Get synced lyrics
    content, metadata = get_synced_lyrics(
        track_name=track_name,
        artist_name=artist_name,
        api_key=api_key,
        album_name=album_name,
        prefer_richsync=args.richsync,
        output_format=args.format,
        target_duration=duration if args.airtable_record else None,
        pick=args.pick,
        musixmatch_track_id=args.musixmatch_track_id,
    )
    
    if content:
        # Save to file
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"\n✅ Saved to: {args.output}")
        print(f"📊 Metadata: {json.dumps(metadata, indent=2)}")
        
        # Save to Airtable if requested
        if args.airtable_record and args.save_to_airtable:
            save_to_airtable(args.airtable_record, content, metadata)
    else:
        print("\n❌ Failed to retrieve synced lyrics")


if __name__ == '__main__':
    main()
