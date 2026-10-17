from loguru import logger


def get_vocab():
    """Get the vocabulary dictionary mapping characters to token IDs"""
    _pad = "$"
    _punctuation = ';:,.!?¡¿—…"«»"" '
    _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

    # Create vocabulary dictionary
    symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
    return {symbol: i for i, symbol in enumerate(symbols)}


# Initialize vocabulary
VOCAB = get_vocab()

# Misaki's eSpeak fallback uses raised schwa; Kokoro's vocabulary has no
# dedicated token, so map it to the nearest supported schwa before inference.
IPA_SUBSTITUTIONS = {"ᵊ": "ə"}


def normalize_phonemes(phonemes: str, warn: bool = True) -> str:
    for source, replacement in IPA_SUBSTITUTIONS.items():
        phonemes = phonemes.replace(source, replacement)

    unmapped = sorted(set(phonemes) - VOCAB.keys())
    if unmapped and warn:
        logger.warning(
            "Unmapped IPA symbols will be omitted by the Kokoro vocabulary: "
            f"{unmapped!r}"
        )
    return phonemes


def tokenize(phonemes: str) -> list[int]:
    """Convert phonemes string to token IDs

    Args:
        phonemes: String of phonemes to tokenize

    Returns:
        List of token IDs
    """
    # Strip phonemes to remove leading/trailing spaces that could cause artifacts
    phonemes = normalize_phonemes(phonemes.strip())
    return [VOCAB[symbol] for symbol in phonemes if symbol in VOCAB]
