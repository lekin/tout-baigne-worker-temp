"""Data models for karaoke QA."""
from dataclasses import dataclass, field, asdict, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, get_origin, get_args, get_type_hints


class SyncStatus(str, Enum):
    PENDING = "PENDING"
    SYNC_VERIFIED = "SYNC_VERIFIED"
    SYNC_FAILED = "SYNC_FAILED"
    SYNC_NEEDS_REVIEW = "SYNC_NEEDS_REVIEW"


class FinalStatus(str, Enum):
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class DiagnosisType(str, Enum):
    GLOBAL_OFFSET = "global_offset"
    PROGRESSIVE_DRIFT = "progressive_drift"
    LOCAL_MISMATCH = "local_mismatch"
    SUSPECTED_WRONG_VERSION = "suspected_wrong_version"
    ALIGNMENT_FAILURE = "alignment_failure"
    GOOD = "good"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class LyricWord:
    id: str
    text: str
    source_start_ms: Optional[float] = None
    source_end_ms: Optional[float] = None
    predicted_start_ms: Optional[float] = None
    predicted_end_ms: Optional[float] = None


@dataclass
class LyricLine:
    id: str
    text: str
    source_start_ms: float
    source_end_ms: Optional[float] = None
    words: List[LyricWord] = field(default_factory=list)
    predicted_start_ms: Optional[float] = None
    predicted_end_ms: Optional[float] = None
    start_error_ms: Optional[float] = None
    end_error_ms: Optional[float] = None


@dataclass
class StructuredLyrics:
    source: str  # 'lrc', 'srt', 'richsync'
    source_track_id: Optional[str]
    language: Optional[str]
    lines: List[LyricLine]


@dataclass
class TimelineTransform:
    lyrics_to_source_offset_ms: float = 0.0
    source_to_karaoke_audio_offset_ms: float = 0.0
    karaoke_audio_to_video_offset_ms: float = 0.0

    def musixmatch_to_source(self, t_ms: float) -> float:
        return t_ms + self.lyrics_to_source_offset_ms

    def source_to_musixmatch(self, t_ms: float) -> float:
        return t_ms - self.lyrics_to_source_offset_ms

    def source_to_final_video(self, t_ms: float) -> float:
        return (
            t_ms
            + self.source_to_karaoke_audio_offset_ms
            + self.karaoke_audio_to_video_offset_ms
        )

    def final_video_to_source(self, t_ms: float) -> float:
        return (
            t_ms
            - self.source_to_karaoke_audio_offset_ms
            - self.karaoke_audio_to_video_offset_ms
        )


@dataclass
class AlignmentResult:
    aligner_name: str
    model_name: str
    settings: Dict[str, Any]
    lyrics: StructuredLyrics
    raw_output_path: Optional[str] = None
    error: Optional[str] = None
    estimated_global_offset_ms: Optional[float] = None


@dataclass
class LineTimingMetrics:
    median_error_ms: Optional[float]
    p90_error_ms: Optional[float]
    max_error_ms: Optional[float]
    mean_error_ms: Optional[float]
    std_error_ms: Optional[float]
    sample_count: int


@dataclass
class WordTimingMetrics:
    coverage: float
    aligned_count: int
    total_count: int
    median_error_ms: Optional[float]
    p90_error_ms: Optional[float]
    max_error_ms: Optional[float]


@dataclass
class SyncMetrics:
    line_start: LineTimingMetrics
    line_end: Optional[LineTimingMetrics]
    word: WordTimingMetrics
    total_expected_lines: int
    line_alignment_coverage: float
    largest_unresolved_lyric_region_ms: float
    drift_slope: float  # predicted_time = source_time * slope + intercept_ms
    drift_intercept_ms: float
    drift_residual_std_ms: float
    duration_mismatch_ms: float
    source_duration_ms: Optional[float]
    predicted_duration_ms: Optional[float]


@dataclass
class SyncDiagnosis:
    type: DiagnosisType
    confidence: Confidence
    description: str
    estimated_global_offset_ms: Optional[float] = None
    problem_ranges: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SyncReport:
    record_id: str
    track_name: str
    artist_name: str
    musixmatch_track_id: Optional[str]
    transform: TimelineTransform
    metrics: SyncMetrics
    diagnosis: SyncDiagnosis
    status: SyncStatus
    lyrics: StructuredLyrics
    alignment_result: AlignmentResult
    gate_results: Dict[str, Any] = field(default_factory=dict)
    cache_key: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    # Benchmark/operational metadata (all optional for backwards compatibility).
    source_duration_ms: Optional[float] = None
    vocal_backend: Optional[str] = None
    vocal_model: Optional[str] = None
    vocal_separation_time_s: Optional[float] = None
    vocal_cache_hit: Optional[bool] = None
    vocal_stem_duration_ms: Optional[float] = None
    vocal_stem_delta_ms: Optional[float] = None
    stable_ts_status: Optional[str] = None
    stable_ts_diagnosis: Optional[str] = None
    stable_ts_median_error_ms: Optional[float] = None
    stable_ts_p90_error_ms: Optional[float] = None
    vocal_path: Optional[str] = None
    audio_path: Optional[str] = None


@dataclass
class QAAttempt:
    source: str  # e.g. 'musixmatch:12345' or 'qa_rebuilt'
    lyrics: StructuredLyrics
    sync_report: Optional[SyncReport] = None


@dataclass
class QAResult:
    record_id: str
    status: FinalStatus
    sync_verified: bool
    attempts: List[QAAttempt] = field(default_factory=list)
    final_source: Optional[str] = None
    correction_applied: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class BenchmarkLabel:
    record_id: str
    manual_status: str  # good, global_offset, local_sync_problem, wrong_version
    problem_ranges: List[Dict[str, float]] = field(default_factory=list)
    known_global_offset_ms: Optional[float] = None
    human_severity: Optional[str] = None  # obvious, subtle
    manual_line_onsets: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    total: int
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    fpr: float
    fnr: float
    per_track: List[Dict[str, Any]]
    details: Optional[Dict[str, Any]] = None


def to_json(obj: Any) -> Any:
    """Convert QA dataclasses/enums to JSON-serializable dicts/lists."""
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: to_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_json(v) for v in obj]
    if isinstance(obj, float) and obj != obj:  # NaN
        return None
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_json(getattr(obj, k)) for k in obj.__dataclass_fields__}
    return obj


T = TypeVar("T")


def _unwrap_optional(cls: Any) -> Any:
    """Return the non-None type from Optional[X] or Union[X, None]."""
    origin = get_origin(cls)
    if origin is Union:
        args = [a for a in get_args(cls) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return cls


def from_json(cls: Type[T], data: Any) -> T:
    """Convert a JSON-serializable dict/list back into a dataclass instance."""
    if data is None:
        return None

    cls = _unwrap_optional(cls)

    if isinstance(cls, type):
        if issubclass(cls, Enum):
            for member in cls:
                if member.value == data:
                    return member
            return None

    origin = get_origin(cls)
    if origin is list or origin is List:
        args = get_args(cls)
        item_cls = args[0] if args else Any
        return [from_json(item_cls, v) for v in data]

    if origin is dict or origin is Dict:
        return {k: v for k, v in data.items()}

    if is_dataclass(cls):
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict for dataclass {cls.__name__}, got {type(data)}")
        kwargs = {}
        for field_name, field_type in get_type_hints(cls).items():
            if field_name in data:
                raw = data[field_name]
                if raw is None:
                    kwargs[field_name] = None
                else:
                    kwargs[field_name] = from_json(field_type, raw)
        return cls(**kwargs)

    return data
