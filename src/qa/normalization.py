"""Lyrics normalization and token mapping for forced-alignment reconciliation."""
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class NormalizedToken:
    """A token after normalization, with a back-pointer to the source lyric."""

    source_text: str
    normalized: str
    source_word_idx: int
    source_line_idx: int
    source_word_id: str


@dataclass
class NormalizedLine:
    """A normalized lyric line with source line index and token list."""

    source_idx: int
    source_line_id: str
    tokens: List[NormalizedToken] = field(default_factory=list)
    raw_text: str = ""


@dataclass
class NormalizedLyrics:
    """A normalized/canonical representation of lyrics, preserving source mapping."""

    source: str
    language: Optional[str]
    lines: List[NormalizedLine]


class LyricsNormalizer:
    """
    Normalize lyrics for fair comparison and alignment while preserving the
    ability to map results back to the original tokens.

    Handles case, whitespace, punctuation, accents, repeated words, ad-libs,
    and parenthesized markers such as '(chorus)'.
    """

    def __init__(
        self,
        lowercase: bool = True,
        strip_accents: bool = True,
        remove_punctuation: bool = True,
        normalize_contractions: bool = True,
        collapse_repeated_words: bool = False,
    ):
        self.lowercase = lowercase
        self.strip_accents = strip_accents
        self.remove_punctuation = remove_punctuation
        self.normalize_contractions = normalize_contractions
        self.collapse_repeated_words = collapse_repeated_words

    def normalize_text(self, text: str) -> str:
        result = text
        if self.lowercase:
            result = result.lower()
        if self.strip_accents:
            result = unicodedata.normalize("NFKD", result)
            result = "".join(c for c in result if not unicodedata.combining(c))
        # Remove parenthesized chorus / section markers (e.g. "(Chorus)").
        result = re.sub(r"\([^)]*\)", "", result)
        # Remove common ad-lib markers and non-lyric tokens.
        result = re.sub(r"\b(yeah[- ]*yeah|oh[- ]*oh|uh[- ]*huh|mm[- ]*mm)\b", "", result, flags=re.IGNORECASE)
        # Normalize contractions if requested.
        if self.normalize_contractions:
            result = self._normalize_contractions(result)
        if self.remove_punctuation:
            result = re.sub(r"[^\w\s'-]", "", result)
        # Collapse repeated words/sections only when requested.
        if self.collapse_repeated_words:
            result = re.sub(r"\b(\w+)(?:\s+\1)+\b", r"\1", result)
        # Collapse whitespace and trim.
        result = re.sub(r"\s+", " ", result).strip()
        return result

    def _normalize_contractions(self, text: str) -> str:
        """Expand or normalize common English contractions."""
        # Simple substitutions that preserve token count where possible.
        replacements = {
            r"\bwon't\b": "will not",
            r"\bcan't\b": "can not",
            r"\bain't\b": "is not",
            r"\bgonna\b": "going to",
            r"\bwanna\b": "want to",
            r"\blemme\b": "let me",
            r"\bgimme\b": "give me",
            r"\b'cause\b": "because",
            r"\bya'll\b": "you all",
            r"\by'all\b": "you all",
        }
        for pattern, repl in replacements.items():
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        # Apostrophe contractions: strip trailing 's, 're, 'll, 've, 'd, n't
        text = re.sub(r"n't\b", " not", text, flags=re.IGNORECASE)
        text = re.sub(r"'re\b", " are", text, flags=re.IGNORECASE)
        text = re.sub(r"'ll\b", " will", text, flags=re.IGNORECASE)
        text = re.sub(r"'ve\b", " have", text, flags=re.IGNORECASE)
        text = re.sub(r"'d\b", " would", text, flags=re.IGNORECASE)
        text = re.sub(r"'s\b", " is", text, flags=re.IGNORECASE)
        # Possessive 's may also split, but for singing-alignment timing this is
        # acceptable. The token mapper can still re-merge small changes.
        return text

    def normalize_line(self, source_idx: int, source_line_id: str, text: str) -> NormalizedLine:
        raw_text = text.strip()
        normalized_text = self.normalize_text(text)
        tokens = [
            NormalizedToken(
                source_text=t,
                normalized=self.normalize_token(t),
                source_word_idx=i,
                source_line_idx=source_idx,
                source_word_id=f"W{source_idx:04d}-{i:04d}",
            )
            for i, t in enumerate(raw_text.split())
        ]
        # If normalization changes token boundaries (e.g. contractions), re-tokenize
        # the normalized text and map back to the closest raw token by index.
        norm_words = normalized_text.split()
        tokens = [
            NormalizedToken(
                source_text=tokens[i].source_text if i < len(tokens) else norm_words[i],
                normalized=norm_words[i],
                source_word_idx=i,
                source_line_idx=source_idx,
                source_word_id=tokens[i].source_word_id if i < len(tokens) else f"W{source_idx:04d}-{i:04d}",
            )
            for i in range(len(norm_words))
        ]
        return NormalizedLine(
            source_idx=source_idx,
            source_line_id=source_line_id,
            tokens=tokens,
            raw_text=raw_text,
        )

    def normalize_token(self, text: str) -> str:
        """Normalize a single token (no splitting)."""
        return self.normalize_text(text)


class TokenMapper:
    """Map predicted alignment tokens back to source lyric words."""

    def __init__(self, normalizer: Optional[LyricsNormalizer] = None):
        self.normalizer = normalizer or LyricsNormalizer()

    def map_predicted_to_source(
        self,
        source_words: List[str],
        predicted_words: List[str],
    ) -> List[Optional[int]]:
        """
        Return a list mapping each predicted token index to a source word index
        (or None if it cannot be matched). The mapping is greedy and sequential.
        """
        norm = self.normalizer
        norm_source = [norm.normalize_token(w) for w in source_words]
        norm_predicted = [norm.normalize_token(w) for w in predicted_words]
        mapping: List[Optional[int]] = []
        src_idx = 0

        for pred in norm_predicted:
            if src_idx >= len(norm_source):
                mapping.append(None)
                continue
            # Try current source token; if it doesn't match, look ahead a few
            # positions for repeated sections or small misalignments.
            match = None
            for lookahead in range(0, min(5, len(norm_source) - src_idx)):
                if pred == norm_source[src_idx + lookahead]:
                    match = src_idx + lookahead
                    break
            if match is not None:
                mapping.append(match)
                src_idx = match + 1
            else:
                mapping.append(None)

        return mapping

    def build_normalized_lyrics(
        self,
        source_lines: List[Tuple[int, str, str]],  # (idx, id, text)
        source_type: str,
        language: Optional[str],
    ) -> NormalizedLyrics:
        """Normalize a list of source lines."""
        lines = [
            self.normalizer.normalize_line(idx, line_id, text)
            for idx, line_id, text in source_lines
        ]
        return NormalizedLyrics(
            source=source_type, language=language, lines=lines
        )
