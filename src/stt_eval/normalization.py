import re
import unicodedata

_PUNCTUATION_RE = re.compile(
    r"[\s"
    r"\u3000"
    r"。？！，、；：「」『』（）《》〈〉【】〔〕—…．"
    r".?!,;:\"'`()\[\]{}<>~\-_/\\|@#$%^&*=+"
    r"]+"
)


def normalize_for_cer(text: str) -> str:
    """Normalize text for Chinese-friendly CER comparison."""

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.lower()
    return _PUNCTUATION_RE.sub("", normalized)
