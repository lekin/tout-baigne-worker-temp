"""Airtable client for managing track records."""
from typing import List, Optional, Dict, Any
from pyairtable import Api
from pyairtable.formulas import match
from src.config import settings
import requests
import os


def _airtable_formula_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


class AirtableClient:
    """Client for interacting with Airtable."""
    
    def __init__(self):
        """Initialize the Airtable client."""
        self.api = Api(settings.airtable_api_key)
        self.table = self.api.table(
            settings.airtable_base_id,
            settings.airtable_table_name
        )

    def get_table(self, table_name: str):
        return self.api.table(settings.airtable_base_id, table_name)
    
    def get_record(self, record_id: str) -> Dict[str, Any]:
        """
        Get a single record by ID.
        
        Args:
            record_id: The Airtable record ID
            
        Returns:
            The record data
        """
        return self.table.get(record_id)
    
    def get_records_by_filters(
        self,
        genres: Optional[List[str]] = None,
        event_types: Optional[List[str]] = None,
        decades: Optional[List[str]] = None,
        max_records: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get records filtered by genre, event type, and/or decade.
        
        Args:
            genres: List of genre names to filter by
            event_types: List of event type IDs to filter by
            decades: List of decade IDs to filter by
            max_records: Maximum number of records to return
            
        Returns:
            List of matching records
        """
        formulas = []
        
        # Build formula for genres (multi-select field)
        if genres:
            genre_conditions = [
                f"FIND({_airtable_formula_string(genre)}, {{Genre}})" for genre in genres
            ]
            formulas.append(f"OR({', '.join(genre_conditions)})")
        
        # Build formula for event types (linked records)
        if event_types:
            event_conditions = [
                f"FIND({_airtable_formula_string(event_type)}, ARRAYJOIN({{Event type}}))" 
                for event_type in event_types
            ]
            formulas.append(f"OR({', '.join(event_conditions)})")
        
        # Build formula for decades (linked records)
        if decades:
            decade_conditions = [
                f"FIND({_airtable_formula_string(decade)}, ARRAYJOIN({{Decade}}))" 
                for decade in decades
            ]
            formulas.append(f"OR({', '.join(decade_conditions)})")
        
        # Combine all formulas with AND
        formula = None
        if formulas:
            if len(formulas) == 1:
                formula = formulas[0]
            else:
                formula = f"AND({', '.join(formulas)})"
        
        return self.table.all(formula=formula, max_records=max_records)

    def get_records_by_filters_for_table(
        self,
        table_name: str,
        genres: Optional[List[str]] = None,
        event_types: Optional[List[str]] = None,
        event_types_field_name: str = "Event type",
        decades: Optional[List[str]] = None,
        min_rating: Optional[float] = None,
        max_records: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        table = self.get_table(table_name)

        formulas = []
        if genres:
            genre_conditions = [
                f"FIND({_airtable_formula_string(genre)}, {{Genre}})" for genre in genres
            ]
            formulas.append(f"OR({', '.join(genre_conditions)})")

        if event_types:
            event_field_ref = "{" + str(event_types_field_name) + "}"
            event_conditions = [
                f"FIND({_airtable_formula_string(event_type)}, ARRAYJOIN({event_field_ref}))"
                for event_type in event_types
            ]
            formulas.append(f"OR({', '.join(event_conditions)})")

        if decades:
            decade_conditions = [
                f"FIND({_airtable_formula_string(decade)}, ARRAYJOIN({{Decade}}))"
                for decade in decades
            ]
            formulas.append(f"OR({', '.join(decade_conditions)})")

        if min_rating is not None:
            try:
                min_rating_value = float(min_rating)
            except Exception:
                min_rating_value = None
            if min_rating_value is not None:
                formulas.append(f"{{Rating}} >= {min_rating_value}")

        formula = None
        if formulas:
            if len(formulas) == 1:
                formula = formulas[0]
            else:
                formula = f"AND({', '.join(formulas)})"

        try:
            return table.all(formula=formula, max_records=max_records)
        except Exception:
            return []

    def get_records_by_tag(
        self,
        table_name: str,
        tag: str,
        max_records: Optional[int] = None,
        tags_field_name: str = "Tags",
    ) -> List[Dict[str, Any]]:
        table = self.get_table(table_name)
        formula = f"FIND({_airtable_formula_string(tag)}, {{{tags_field_name}}})"
        try:
            return table.all(formula=formula, max_records=max_records)
        except Exception:
            return []
    
    def update_record(
        self,
        record_id: str,
        fields: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update a record with new field values.
        
        Args:
            record_id: The Airtable record ID
            fields: Dictionary of fields to update
            
        Returns:
            The updated record data
        """
        try:
            return self.table.update(record_id, fields)
        except Exception as e:
            print(f"⚠️  Warning: Could not update Airtable: {e}")
            return {}
    
    def get_video_url(self, record: Dict[str, Any]) -> Optional[str]:
        """
        Extract video URL from a record.
        
        Args:
            record: The Airtable record
            
        Returns:
            The video URL if present, None otherwise
        """
        fields = record.get("fields", {})
        
        # Try multiple field name variations
        video_field = (
            fields.get("Video URL") or 
            fields.get("Video") or
            fields.get("YouTube URL (from Music videos)") or
            fields.get("Open music video URL") or
            fields.get("YouTube URL") or
            fields.get("Youtube URL") or
            fields.get("YouTube") or
            fields.get("YT URL") or
            fields.get("YT")
        )
        
        if isinstance(video_field, list) and len(video_field) > 0:
            # Handle list of URLs
            if isinstance(video_field[0], dict):
                return video_field[0].get("url")
            elif isinstance(video_field[0], str):
                return video_field[0]
        elif isinstance(video_field, str):
            return video_field

        # Prefer Airtable attachments that look like video files.
        video_exts = ('.mp4', '.mov', '.mkv', '.webm', '.m4v')
        for v in fields.values():
            if not isinstance(v, list):
                continue
            for item in v:
                if not isinstance(item, dict):
                    continue
                url = item.get('url')
                if not isinstance(url, str) or not url:
                    continue
                mime = item.get('type') or item.get('mimeType')
                filename = item.get('filename')
                if isinstance(mime, str) and mime.lower().startswith('video/'):
                    return url
                if isinstance(filename, str) and filename.lower().endswith(video_exts):
                    return url

        found = self._find_first_url_in_fields(fields)
        if not found:
            return None

        s = found.lower()
        if (
            'youtube.com' in s
            or 'youtu.be' in s
            or 'drive.google.com' in s
            or 'airtableusercontent.com' in s
            or 'dl.airtable.com' in s
            or s.endswith('.mp4')
            or s.endswith('.mov')
            or s.endswith('.mkv')
            or s.endswith('.webm')
            or s.endswith('.m3u8')
        ):
            return found

        return None

    def get_gdrive_music_video_url(self, record: Dict[str, Any]) -> Optional[str]:
        fields = record.get("fields", {})
        video_field = None
        for key in [
            "GDrive videos (from Music videos)",
            "GDrive videos (from Tracks)",
            "GDrive videos",
            "GDrive video",
            "GDrive",
            "Google Drive videos",
            "Google Drive video",
            "Drive videos",
            "Drive video",
        ]:
            if key in fields and fields.get(key):
                video_field = fields.get(key)
                break

        url: Optional[str] = None
        if isinstance(video_field, list) and len(video_field) > 0:
            if isinstance(video_field[0], dict):
                url = video_field[0].get("url")
            elif isinstance(video_field[0], str):
                first = video_field[0]
                if isinstance(first, str) and first.startswith("rec"):
                    for linked_id in [v for v in video_field if isinstance(v, str)]:
                        resolved = self._resolve_music_video_asset_url(linked_id)
                        if resolved:
                            url = resolved
                            break
                else:
                    url = first
        elif isinstance(video_field, str):
            url = video_field

        if not url:
            return None

        try:
            lower = str(url).lower()
            if 'youtube.com' in lower or 'youtu.be' in lower:
                return None
        except Exception:
            pass

        if isinstance(url, str) and url.startswith("rec"):
            resolved = self._resolve_music_video_asset_url(url)
            if not resolved:
                return None
            try:
                lower = str(resolved).lower()
                if 'youtube.com' in lower or 'youtu.be' in lower:
                    return None
            except Exception:
                pass
            return resolved

        if "drive.google.com" not in url:
            return url

        try:
            if "/file/d/" in url:
                file_id = url.split("/file/d/")[1].split("/")[0]
            elif "id=" in url:
                file_id = url.split("id=")[1].split("&")[0]
            else:
                return url
            return f"https://drive.google.com/uc?export=download&id={file_id}"
        except Exception:
            return url

    def _resolve_music_video_asset_url(self, record_id: str) -> Optional[str]:
        table_names = [
            "Music videos",
            "Music Videos",
            "Music video",
            "Music Video",
        ]

        for name in table_names:
            try:
                table = self.api.table(settings.airtable_base_id, name)
                linked = table.get(record_id)
            except Exception:
                continue

            fields = linked.get("fields", {}) if isinstance(linked, dict) else {}
            found_drive = None
            try:
                for v in fields.values():
                    if isinstance(v, str):
                        if 'drive.google.com' in v.lower():
                            found_drive = v
                            break
                    if isinstance(v, dict):
                        u = v.get('url')
                        if isinstance(u, str) and 'drive.google.com' in u.lower():
                            found_drive = u
                            break
                    if isinstance(v, list):
                        for item in v:
                            if isinstance(item, str) and 'drive.google.com' in item.lower():
                                found_drive = item
                                break
                            if isinstance(item, dict):
                                u = item.get('url')
                                if isinstance(u, str) and 'drive.google.com' in u.lower():
                                    found_drive = u
                                    break
                        if found_drive:
                            break
            except Exception:
                found_drive = None

            found = found_drive or self._find_first_url_in_fields(fields)
            if found:
                return found

        return None

    def _find_first_url_in_fields(self, fields: Dict[str, Any]) -> Optional[str]:
        def walk(value: Any) -> Optional[str]:
            if isinstance(value, str):
                if value.startswith("http://") or value.startswith("https://"):
                    return value
                return None
            if isinstance(value, dict):
                u = value.get("url")
                if isinstance(u, str) and (u.startswith("http://") or u.startswith("https://")):
                    return u
                for v in value.values():
                    got = walk(v)
                    if got:
                        return got
                return None
            if isinstance(value, list):
                for item in value:
                    got = walk(item)
                    if got:
                        return got
            return None

        preferred = [
            "GDrive videos",
            "GDrive video",
            "GDrive",
            "Video file",
            "Video",
            "File",
            "URL",
            "Link",
        ]

        for key in preferred:
            if key in fields:
                got = walk(fields.get(key))
                if got:
                    return got

        for v in fields.values():
            got = walk(v)
            if got:
                return got

        return None
    
    def get_lyrics_srt(self, record: Dict[str, Any]) -> Optional[str]:
        """
        Extract lyrics SRT content from a record.
        Prioritizes Genius-synced SRT, then Whisper SRT, then legacy fields.
        
        Args:
            record: The Airtable record
            
        Returns:
            The SRT content if present, None otherwise
        """
        fields = record.get("fields", {})
        # Priority: Genius > Whisper > Legacy
        return fields.get("SRT (Genius)") or fields.get("SRT (Whisper)") or fields.get("SRT") or fields.get("Lyrics SRT") or fields.get("Lyrics")
    
    def get_whisper_srt(self, record: Dict[str, Any]) -> Optional[str]:
        """
        Get Whisper-generated SRT specifically.
        
        Args:
            record: The Airtable record
            
        Returns:
            The Whisper SRT content if present, None otherwise
        """
        fields = record.get("fields", {})
        return fields.get("SRT (Whisper)") or fields.get("SRT")
    
    def get_genius_srt(self, record: Dict[str, Any]) -> Optional[str]:
        """
        Get Genius-synced SRT specifically.
        
        Args:
            record: The Airtable record
            
        Returns:
            The Genius SRT content if present, None otherwise
        """
        fields = record.get("fields", {})
        return fields.get("SRT (Genius)")
    
    def get_audio_to_video_offset(self, record: Dict[str, Any]) -> float:
        """
        Get the audio to music video time offset.
        This represents how much the video is delayed compared to the audio.
        
        Args:
            record: The Airtable record
            
        Returns:
            The offset in seconds (0 if not set)
        """
        fields = record.get("fields", {})
        offset = fields.get("Audio to music video time offset (s)", 0)
        return float(offset) if offset else 0.0
    
    def get_time_offset(self, record: Dict[str, Any]) -> float:
        """
        Extract time offset from a record.
        
        Args:
            record: The Airtable record
            
        Returns:
            The time offset in seconds (default 0.0)
        """
        fields = record.get("fields", {})
        # Try both field name variations
        offset = fields.get("Music video singing time offset (s)") or fields.get("Music video singing time offset", 0.0)
        try:
            return float(offset) if offset else 0.0
        except (ValueError, TypeError):
            return 0.0
    
    def get_vocal_file_url(self, record: Dict[str, Any]) -> Optional[str]:
        """
        Extract vocal file URL from a record.
        
        Args:
            record: The Airtable record
            
        Returns:
            The vocal file URL if present, None otherwise
        """
        fields = record.get("fields", {})
        vocal_field = fields.get("Vocal") or fields.get("Vocals")
        
        if isinstance(vocal_field, list) and len(vocal_field) > 0:
            # Airtable attachments are lists of dicts
            if isinstance(vocal_field[0], dict):
                return vocal_field[0].get("url")
        
        return None
    
    def get_video_file_url(self, record: Dict[str, Any]) -> Optional[str]:
        """
        Extract video file URL from a record.
        
        Args:
            record: The Airtable record
            
        Returns:
            The video file URL if present, None otherwise
        """
        fields = record.get("fields", {})
        video_field = fields.get("Video file") or fields.get("Video File")
        
        if isinstance(video_field, list) and len(video_field) > 0:
            # Airtable attachments are lists of dicts
            if isinstance(video_field[0], dict):
                return video_field[0].get("url")
        
        return None
    
    def upload_video_file(self, record_id: str, video_path: str) -> bool:
        """
        Upload a video file to the 'Video file' attachment field in Airtable.
        
        Args:
            record_id: The Airtable record ID
            video_path: Path to the video file to upload
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from pathlib import Path
            import os
            
            if not os.path.exists(video_path):
                print(f"Error: Video file not found at {video_path}")
                return False
            
            filename = Path(video_path).name
            
            # Upload file using pyairtable's attachment upload
            # Read file content and upload
            with open(video_path, 'rb') as f:
                file_content = f.read()
                
            # Use pyairtable's attachment format: list of tuples (filename, file_content, mime_type)
            self.table.update(record_id, {
                'Video file': [
                    (filename, file_content, 'video/mp4')
                ]
            })
            
            return True
        except Exception as e:
            print(f"Error uploading video: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_audio_file_url(self, record: Dict[str, Any]) -> Optional[str]:
        """
        Extract audio file URL from a record (original audio track).
        Prioritizes Google Drive link over attachment.
        
        Args:
            record: The Airtable record
            
        Returns:
            The audio file URL if present, None otherwise
        """
        fields = record.get("fields", {})
        
        # First check for Google Drive link
        gdrive_link = fields.get("Link (from GDrive Audio files)")
        if gdrive_link:
            # Handle if it's a list (from lookup field)
            if isinstance(gdrive_link, list) and len(gdrive_link) > 0:
                gdrive_link = gdrive_link[0]
            
            # Convert Google Drive sharing link to direct download link
            if "drive.google.com" in gdrive_link:
                # Extract file ID from various Google Drive URL formats
                if "/file/d/" in gdrive_link:
                    file_id = gdrive_link.split("/file/d/")[1].split("/")[0]
                elif "id=" in gdrive_link:
                    file_id = gdrive_link.split("id=")[1].split("&")[0]
                else:
                    return gdrive_link  # Return as-is if format unknown
                
                # Return direct download link
                return f"https://drive.google.com/uc?export=download&id={file_id}"
            return gdrive_link
        
        # Fallback to Audio file attachment
        audio_field = fields.get("Audio file") or fields.get("Audio File")
        
        if isinstance(audio_field, list) and len(audio_field) > 0:
            # Airtable attachments are lists of dicts
            if isinstance(audio_field[0], dict):
                return audio_field[0].get("url")
        
        return None
    
    def upload_file_to_airtable(self, record_id: str, file_path: str, field_name: str) -> Dict[str, Any]:
        """
        Upload a file to Airtable by first uploading to a temporary hosting service.
        
        For now, this just logs the file path. In production, you would:
        1. Upload to S3/GCS/Cloudinary
        2. Get public URL
        3. Update Airtable with URL
        
        Args:
            record_id: The Airtable record ID
            file_path: Path to the file to upload
            field_name: Airtable field name (e.g., "Vocal", "Video file")
            
        Returns:
            The updated record data
        """
        import os
        
        if not os.path.exists(file_path):
            print(f"⚠️  File not found: {file_path}")
            return {}
        
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        
        print(f"ℹ️  {field_name}: {filename} ({file_size:.1f} MB)")
        print(f"ℹ️  To upload to Airtable: manually add to '{field_name}' field")
        
        return {}
    
    def save_srt_only(self, record_id: str, srt_content: str, srt_type: str = "whisper") -> Dict[str, Any]:
        """
        Save only SRT content to Airtable (for progressive updates).
        
        Args:
            record_id: The Airtable record ID
            srt_content: The SRT content to save
            srt_type: Type of SRT - "whisper" or "genius" (default: "whisper")
            
        Returns:
            The updated record data
        """
        field_name = "SRT (Genius)" if srt_type == "genius" else "SRT (Whisper)"
        return self.update_record(record_id, {field_name: srt_content})
    
    def save_offset_only(self, record_id: str, time_offset: float) -> Dict[str, Any]:
        """
        Save only time offset to Airtable (for progressive updates).
        
        Args:
            record_id: The Airtable record ID
            time_offset: The time offset in seconds
            
        Returns:
            The updated record data
        """
        return self.update_record(record_id, {
            "Music video singing time offset (s)": time_offset
        })
    
    def save_srt_and_offset(self, record_id: str, srt_content: str, time_offset: float, vocal_url: Optional[str] = None, srt_type: str = "whisper") -> Dict[str, Any]:
        """
        Save SRT content, time offset, and optionally vocal file URL to Airtable.
        
        Args:
            record_id: The Airtable record ID
            srt_content: The SRT content to save
            time_offset: The detected time offset in seconds
            vocal_url: Optional URL to the vocal file (for Airtable attachment)
            srt_type: Type of SRT - "whisper" or "genius" (default: "whisper")
            
        Returns:
            The updated record data
        """
        fields = {
            "Music video singing time offset (s)": time_offset
        }
        
        # Save to appropriate SRT field
        if srt_type == "genius":
            fields["SRT (Genius)"] = srt_content
        else:
            fields["SRT (Whisper)"] = srt_content
        
        # Add vocal attachment if URL provided
        if vocal_url:
            fields["Vocal"] = [{"url": vocal_url}]
        
        return self.update_record(record_id, fields)
    
    def upload_file_to_airtable(self, file_path: str, record_id: str, field_name: str) -> bool:
        """
        Upload a file to Airtable using direct upload API.
        
        Args:
            file_path: Path to the local file
            record_id: The Airtable record ID
            field_name: The field name to upload to (e.g., "Video file", "Vocal")
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                return False
            
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            filename = os.path.basename(file_path)
            
            print(f"📤 Uploading {filename} ({file_size:.1f} MB) to Airtable...")
            
            # Airtable supports direct file uploads via multipart/form-data
            # Step 1: Upload file to Airtable's upload endpoint
            upload_url = f"https://content.airtable.com/v0/{settings.airtable_base_id}/{settings.airtable_table_name}/{record_id}/{field_name}/uploadAttachment"
            
            headers = {
                "Authorization": f"Bearer {settings.airtable_api_key}",
            }
            
            with open(file_path, 'rb') as f:
                files = {
                    'file': (filename, f, 'application/octet-stream')
                }
                
                response = requests.post(
                    upload_url,
                    headers=headers,
                    files=files,
                    timeout=300  # 5 minutes timeout for large files
                )
            
            if response.status_code in [200, 201]:
                print(f"✅ File uploaded to Airtable successfully")
                return True
            else:
                # Fallback: Try using URL-based attachment
                print(f"⚠️  Direct upload not available, using URL method...")
                return self._upload_via_tmpfiles(file_path, record_id, field_name)
                
        except Exception as e:
            print(f"⚠️  Upload error: {e}, trying fallback...")
            return self._upload_via_tmpfiles(file_path, record_id, field_name)
    
    def _upload_via_tmpfiles(self, file_path: str, record_id: str, field_name: str) -> bool:
        """
        Fallback: Upload via tmpfiles.org (simple, no account needed).
        
        Args:
            file_path: Path to the local file
            record_id: The Airtable record ID
            field_name: The field name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            filename = os.path.basename(file_path)
            
            # Use tmpfiles.org - simple, no API key needed
            with open(file_path, 'rb') as f:
                response = requests.post(
                    'https://tmpfiles.org/api/v1/upload',
                    files={'file': (filename, f)},
                    timeout=300
                )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'success':
                    # tmpfiles.org returns URL in format: https://tmpfiles.org/123/file.mp4
                    # We need to convert to direct download: https://tmpfiles.org/dl/123/file.mp4
                    file_url = result.get('data', {}).get('url', '')
                    if file_url:
                        # Convert to direct download link
                        file_url = file_url.replace('tmpfiles.org/', 'tmpfiles.org/dl/')
                        print(f"✓ File uploaded to temporary host: {file_url}")
                        
                        # Attach to Airtable record
                        self.update_record(record_id, {
                            field_name: [{"url": file_url}]
                        })
                        
                        print(f"✅ File attached to Airtable")
                        return True
            
            print(f"❌ Upload failed")
            return False
                
        except Exception as e:
            print(f"❌ Fallback upload failed: {e}")
            return False
    
    def upload_video_file(self, file_path: str, record_id: str) -> bool:
        """
        Upload a video file to Airtable.
        
        Args:
            file_path: Path to the video file
            record_id: The Airtable record ID
            
        Returns:
            True if successful, False otherwise
        """
        return self.upload_file_to_airtable(file_path, record_id, "Video file")
    
    def upload_attachment_to_record(
        self,
        table_name: str,
        record_id: str,
        field_name: str,
        file_path: str,
        content_type: str = "video/mp4",
    ) -> bool:
        """Upload a local file to an Airtable attachment field.

        Uses Airtable's direct upload API (base64, up to 5MB).
        Falls back to multipart upload for larger files and tmpfiles.org
        as a last resort.
        """
        import base64
        import os

        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return False

        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        print(f"📤 Uploading {filename} ({file_size / (1024 * 1024):.1f} MB) to Airtable...")

        # Method 1: base64 JSON upload (up to 5 MB)
        if file_size <= 5 * 1024 * 1024:
            try:
                with open(file_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")

                url = f"https://content.airtable.com/v0/{settings.airtable_base_id}/{record_id}/{field_name}/uploadAttachment"
                headers = {
                    "Authorization": f"Bearer {settings.airtable_api_key}",
                    "Content-Type": "application/json",
                }
                body = {
                    "contentType": content_type,
                    "file": b64,
                    "filename": filename,
                }
                resp = requests.post(url, headers=headers, json=body, timeout=120)
                if resp.status_code in (200, 201):
                    print(f"✅ Uploaded {filename} to Airtable")
                    return True
                print(f"⚠️  Base64 upload failed: {resp.status_code} {resp.text[:200]}")
            except Exception as e:
                print(f"⚠️  Base64 upload error: {e}")

        # Method 2: multipart direct upload
        try:
            url = f"https://content.airtable.com/v0/{settings.airtable_base_id}/{record_id}/{field_name}/uploadAttachment"
            headers = {"Authorization": f"Bearer {settings.airtable_api_key}"}
            with open(file_path, "rb") as f:
                files = {"file": (filename, f, content_type)}
                resp = requests.post(url, headers=headers, files=files, timeout=300)
            if resp.status_code in (200, 201):
                print(f"✅ Uploaded {filename} to Airtable")
                return True
            print(f"⚠️  Multipart upload failed: {resp.status_code} {resp.text[:200]}")
        except Exception as e:
            print(f"⚠️  Multipart upload error: {e}")

        # Method 3: fallback to temporary public URL
        print("⚠️  Falling back to temporary file hosting...")
        return self._upload_via_tmpfiles_to_table(table_name, record_id, field_name, file_path)

    def _upload_via_tmpfiles_to_table(self, table_name: str, record_id: str, field_name: str, file_path: str) -> bool:
        """Fallback upload to Airtable via tmpfiles.org public URL."""
        try:
            filename = os.path.basename(file_path)
            with open(file_path, "rb") as f:
                resp = requests.post(
                    "https://tmpfiles.org/api/v1/upload",
                    files={"file": (filename, f)},
                    timeout=300,
                )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "success":
                    file_url = data.get("data", {}).get("url", "").replace("tmpfiles.org/", "tmpfiles.org/dl/")
                    if file_url:
                        self.get_table(table_name).update(record_id, {field_name: [{"url": file_url, "filename": filename}]})
                        print(f"✅ Attached {filename} via tmpfiles")
                        return True
            print(f"❌ tmpfiles fallback failed: {resp.status_code} {resp.text[:200]}")
        except Exception as e:
            print(f"❌ tmpfiles fallback error: {e}")
        return False

    def upload_vocal_file(self, file_path: str, record_id: str) -> bool:
        """
        Upload a vocal file to Airtable.
        
        Args:
            file_path: Path to the vocal file
            record_id: The Airtable record ID
            
        Returns:
            True if successful, False otherwise
        """
        return self.upload_file_to_airtable(file_path, record_id, "Vocal")
    
    def upload_audio_file(self, file_path: str, record_id: str) -> bool:
        """
        Upload an audio file to Airtable.
        
        Args:
            file_path: Path to the audio file
            record_id: The Airtable record ID
            
        Returns:
            True if successful, False otherwise
        """
        return self.upload_file_to_airtable(file_path, record_id, "Audio file")
