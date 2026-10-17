from dataclasses import dataclass

@dataclass
class WordTimestamp:
    word: str
    start_time: float
    end_time: float


@dataclass
class NormalizationOptions:
    normalize: bool = True
    unit_normalization: bool = False
    url_normalization: bool = True
    email_normalization: bool = True
    optional_pluralization_normalization: bool = True
    phone_normalization: bool = True
    caps_normalization: bool = True
    replace_remaining_symbols: bool = True
    remove_emoji: bool = False
