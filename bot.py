import os
import json
import html
import time
import logging
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
import pykakasi
from janome.tokenizer import Tokenizer as JanomeTokenizer
from pypinyin import pinyin, Style
from unidecode import unidecode
from gtts import gTTS
import cyrtranslit
from korean_romanizer.romanizer import Romanizer as KoreanRomanizer
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate as indic_transliterate

DetectorFactory.seed = 0  # make langdetect deterministic

load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Shortcut command -> Google Translate language code
LANGUAGES = {
    "jap": "ja", "japanese": "ja",
    "kr": "ko", "korean": "ko",
    "zh": "zh-CN", "chinese": "zh-CN",
    "fr": "fr", "french": "fr",
    "es": "es", "spanish": "es",
    "de": "de", "german": "de",
    "it": "it", "italian": "it",
    "pt": "pt", "portuguese": "pt",
    "ru": "ru", "russian": "ru",
    "ar": "ar", "arabic": "ar",
    "hi": "hi", "hindi": "hi",
    "th": "th", "thai": "th",
    "vi": "vi", "vietnamese": "vi",
    "id": "id", "indonesian": "id",
    "tl": "tl", "tagalog": "tl",
    "ms": "ms", "malay": "ms",
}

# Quick-tap buttons shown under plain English messages
QUICK_LANGS = [
    ("🇯🇵 Japanese", "ja"),
    ("🇰🇷 Korean", "ko"),
    ("🇨🇳 Chinese", "zh-CN"),
    ("🇫🇷 French", "fr"),
    ("🇪🇸 Spanish", "es"),
    ("🇩🇪 German", "de"),
    ("🇷🇺 Russian", "ru"),
    ("🇻🇳 Vietnamese", "vi"),
]

# In-memory cache of recent message text, so callback buttons know what to translate.
# Keyed by message_id (str). Cleared on restart -- old buttons just stop working, which
# is fine since the person can simply resend the text.
TEXT_CACHE = {}
MAX_CACHE_ENTRIES = 500

# Per-user default target language, persisted to disk so it survives restarts.
DEFAULTS_FILE = Path(__file__).parent / "user_defaults.json"


def _load_json(path: Path):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Could not read {path}: {e}")
    return {}


def _save_json(path: Path, data):
    try:
        path.write_text(json.dumps(data), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Could not write {path}: {e}")


USER_DEFAULTS = _load_json(DEFAULTS_FILE)   # {str(chat_id): lang_code}

_kks = pykakasi.kakasi()
_janome = JanomeTokenizer()

# は/へ change pronunciation when used as grammatical particles (topic marker / direction
# marker), and a few common greetings keep the old "wa" pronunciation as a fixed idiom
# even though they aren't tagged as particles. Plain kana-by-kana romanization gets this
# wrong (e.g. "koreha" instead of "kore wa"), so these are handled as special cases.
JA_WA_EXCEPTIONS = {
    "こんにちは": "konnichiwa",
    "こんばんは": "konbanwa",
    "今日は": "konnichiwa",
    "今晩は": "konbanwa",
}
# Full-width CJK punctuation -> plain ASCII, shared between Japanese and Chinese
# romanization so punctuation attaches cleanly instead of getting its own stray space.
CJK_PUNCT_MAP = {
    "、": ",", "，": ",", "。": ".", "！": "!", "？": "?",
    "「": '"', "」": '"', "『": '"', "』": '"',
    "；": ";", "：": ":", "（": "(", "）": ")",
    "《": '"', "》": '"', "—": "-", "…": "...",
    "“": '"', "”": '"', "‘": "'", "’": "'",
}


def romanize_japanese(text: str) -> str:
    """Convert Japanese text to Hepburn romaji, correctly handling は/へ as particles."""
    parts = []
    for token in _janome.tokenize(text):
        surface = token.surface
        pos = token.part_of_speech
        if surface in JA_WA_EXCEPTIONS:
            romaji = JA_WA_EXCEPTIONS[surface]
        elif surface == "は" and pos.startswith("助詞"):
            romaji = "wa"
        elif surface == "へ" and pos.startswith("助詞"):
            romaji = "e"
        elif surface in CJK_PUNCT_MAP:
            if parts:
                parts[-1] += CJK_PUNCT_MAP[surface]
            else:
                parts.append(CJK_PUNCT_MAP[surface])
            continue
        else:
            romaji = "".join(item["hepburn"] for item in _kks.convert(surface))
        parts.append(romaji)
    return " ".join(parts)


def romanize_chinese(text: str) -> str:
    """Convert Chinese text to pinyin, attaching punctuation without a stray leading space."""
    parts = []
    for syllable_group in pinyin(text, style=Style.TONE):
        token = syllable_group[0]
        if token in CJK_PUNCT_MAP:
            mapped = CJK_PUNCT_MAP[token]
            if parts:
                parts[-1] += mapped
            else:
                parts.append(mapped)
        else:
            parts.append(token)
    return " ".join(parts)


def get_pronunciation(text: str, lang_code: str):
    """Best-effort romanization/pronunciation for non-Latin scripts."""
    try:
        if lang_code == "ja":
            return romanize_japanese(text)
        if lang_code.startswith("zh"):
            return romanize_chinese(text)
        if lang_code == "ko":
            return KoreanRomanizer(text).romanize()
        if lang_code == "ru":
            return cyrtranslit.to_latin(text, "ru")
        if lang_code == "hi":
            return indic_transliterate(text, sanscript.DEVANAGARI, sanscript.IAST)
        if lang_code in ("ar", "th"):
            # No good lightweight pure-Python transliterator for these scripts (proper
            # ones need heavy ML vowelization models), so this is a rough approximation
            # only -- readable as a rough guide, not phonetically precise.
            return unidecode(text)
        if lang_code in ("el", "he", "uk", "bg", "sr"):
            return unidecode(text)
    except Exception as e:
        logger.warning(f"Pronunciation generation failed: {e}")
    return None


def translate_with_retry(text: str, source: str, target: str, max_attempts: int = 3, delay: float = 1.5) -> str:
    """Call GoogleTranslator, retrying a couple of times on transient failures
    (e.g. brief rate-limiting) before giving up."""
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return GoogleTranslator(source=source, target=target).translate(text)
        except Exception as e:
            last_error = e
            logger.warning(f"Translation attempt {attempt}/{max_attempts} failed: {e}")
            if attempt < max_attempts:
                time.sleep(delay)
    raise last_error
    """Generate a spoken-audio file for text in lang_code. Returns a filepath or None."""
    try:
        tts_lang = lang_code.split("-")[0] if lang_code not in ("zh-CN", "zh-TW") else lang_code
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.close()
        gTTS(text=text, lang=tts_lang).save(tmp.name)
        return tmp.name
    except Exception as e:
        logger.warning(f"Voice generation failed: {e}")
        return None


def cache_text(message_id: int, text: str):
    if len(TEXT_CACHE) > MAX_CACHE_ENTRIES:
        # drop oldest entries to keep memory bounded
        for old_key in list(TEXT_CACHE.keys())[: MAX_CACHE_ENTRIES // 2]:
            TEXT_CACHE.pop(old_key, None)
    TEXT_CACHE[str(message_id)] = text


def quick_lang_keyboard(message_id: int) -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(label, callback_data=f"quick:{code}:{message_id}")
        for label, code in QUICK_LANGS
    ]
    rows = [buttons[i:i + 2] for i in range(0, len(buttons), 2)]
    return InlineKeyboardMarkup(rows)


def default_lang_keyboard() -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(label, callback_data=f"setdefault:{code}")
        for label, code in QUICK_LANGS
    ]
    rows = [buttons[i:i + 2] for i in range(0, len(buttons), 2)]
    return InlineKeyboardMarkup(rows)


def extract_target_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get the text to translate: command arguments, or -- if the command was sent as
    a reply with no arguments -- the text of the message being replied to."""
    if context.args:
        return " ".join(context.args)
    reply = update.message.reply_to_message
    if reply and reply.text:
        return reply.text
    return None


async def send_translation(reply_target, text: str, lang_code: str, label: str):
    """Shared logic: show a status message, translate text into lang_code, then edit
    the status into a nicely formatted result (+ pronunciation + voice clip)."""
    status = await reply_target.reply_text("⏳ Translating...")

    try:
        translated = translate_with_retry(text, source="auto", target=lang_code)
    except Exception as e:
        logger.error(f"Translation error: {e}")
        await status.edit_text("⚠️ Sorry, translation failed. Please try again.")
        return

    safe_translated = html.escape(translated)
    reply = f"🌐 <b>Translation</b>\n{safe_translated}"

    pronunciation = get_pronunciation(translated, lang_code)
    if pronunciation:
        safe_pron = html.escape(pronunciation)
        reply += f"\n\n🔤 <b>Pronunciation</b>\n<i>{safe_pron}</i>"

    await status.edit_text(reply, parse_mode="HTML")

    voice_path = generate_voice(translated, lang_code)
    if voice_path:
        try:
            with open(voice_path, "rb") as f:
                await reply_target.reply_audio(audio=f, title=f"{label} pronunciation")
        except Exception as e:
            logger.warning(f"Sending voice failed: {e}")
        finally:
            os.remove(voice_path)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "👋 Hi! I'm a translation bot.\n\n"
        "• Send me any message in another language and I'll translate it to English automatically.\n"
        "• Send plain English text and I'll show quick-tap buttons to translate it into popular languages.\n"
        "• Use a language command any time, e.g.:\n"
        "   /jap hello  → Japanese translation + pronunciation + voice clip\n"
        "   /fr hello, /es hello, /kr hello, /zh hello, etc.\n"
        "• Reply to any message with a language command (no text needed) to translate "
        "that message, e.g. reply to a message with just /jap.\n\n"
        "• /setdefault → pick a default language once.\n"
        "• /d hello → translate straight into your saved default (also works as a reply).\n\n"
        "Type /langs to see all supported shortcut commands."
    )
    await update.message.reply_text(text)


async def list_langs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    names = sorted(set(LANGUAGES.keys()))
    await update.message.reply_text("Supported commands: " + ", ".join(f"/{n}" for n in names))


async def set_default(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🌐 <b>Pick your default language:</b>",
        parse_mode="HTML",
        reply_markup=default_lang_keyboard(),
    )


async def translate_to_default(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    lang_code = USER_DEFAULTS.get(chat_id)
    if not lang_code:
        await update.message.reply_text(
            "You haven't set a default language yet. Use /setdefault to pick one first."
        )
        return

    text = extract_target_text(update, context)
    if not text:
        await update.message.reply_text(
            "Usage: /d <text to translate>, or reply to a message with /d."
        )
        return

    await send_translation(update.message, text, lang_code, label="/d")


async def translate_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    command = update.message.text.split()[0][1:].split("@")[0].lower()
    lang_code = LANGUAGES.get(command)
    if not lang_code:
        return

    text = extract_target_text(update, context)
    if not text:
        await update.message.reply_text(
            f"Usage: /{command} <text to translate>, or reply to a message with /{command}."
        )
        return

    await send_translation(update.message, text, lang_code, label=f"/{command}")


async def auto_translate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    if not text:
        return

    try:
        detected = detect(text)
    except Exception:
        return  # can't reliably detect (e.g. too short) -> ignore

    if detected == "en":
        # Plain English -> offer quick-tap translation buttons instead of doing nothing.
        cache_text(update.message.message_id, text)
        await update.message.reply_text(
            "🔽 <b>Translate to:</b>",
            parse_mode="HTML",
            reply_markup=quick_lang_keyboard(update.message.message_id),
        )
        return

    status = await update.message.reply_text("⏳ Translating to English...")
    try:
        translated = translate_with_retry(text, source="auto", target="en")
    except Exception as e:
        logger.error(f"Auto-translate error: {e}")
        await status.edit_text("⚠️ Sorry, translation failed. Please try again.")
        return

    safe_original = html.escape(text)
    safe_translated = html.escape(translated)
    reply = f"📝 <b>Original</b>\n{safe_original}"

    pronunciation = get_pronunciation(text, detected)
    if pronunciation:
        safe_pron = html.escape(pronunciation)
        reply += f"\n🔤 <b>Pronunciation</b>\n<i>{safe_pron}</i>"

    reply += f"\n\n🇬🇧 <b>English</b>\n{safe_translated}"
    await status.edit_text(reply, parse_mode="HTML")


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()  # stop the loading spinner on the button
    data = query.data or ""

    if data.startswith("setdefault:"):
        _, lang_code = data.split(":", 1)
        chat_id = str(query.message.chat_id)
        USER_DEFAULTS[chat_id] = lang_code
        _save_json(DEFAULTS_FILE, USER_DEFAULTS)
        label = next((l for l, c in QUICK_LANGS if c == lang_code), lang_code)
        await query.edit_message_text(
            f"✅ Default language set to <b>{html.escape(label)}</b>. Use /d &lt;text&gt; any time.",
            parse_mode="HTML",
        )
        return

    if data.startswith("quick:"):
        _, lang_code, message_id = data.split(":", 2)
        text = TEXT_CACHE.get(message_id)
        if not text:
            await query.message.reply_text(
                "That button expired — please resend the text and try again."
            )
            return
        label = next((l for l, c in QUICK_LANGS if c == lang_code), lang_code)
        await send_translation(query.message, text, lang_code, label=label)
        return


def main():
    if not BOT_TOKEN:
        raise SystemExit("Missing TELEGRAM_BOT_TOKEN. Set it in your .env file.")

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", start))
    app.add_handler(CommandHandler("langs", list_langs))
    app.add_handler(CommandHandler("setdefault", set_default))
    app.add_handler(CommandHandler("d", translate_to_default))

    for cmd in LANGUAGES:
        app.add_handler(CommandHandler(cmd, translate_command))

    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, auto_translate))

    logger.info("Bot starting...")
    app.run_polling()


if __name__ == "__main__":
    main()
