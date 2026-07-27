import streamlit as st
import fitz
from docx import Document
import torch
import html
import os
import re
import subprocess
import tempfile
import wave
from html.parser import HTMLParser
from urllib.error import URLError, HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe
from youtube_transcript_api import YouTubeTranscriptApi

from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import pipeline
from transformers.utils import logging as transformers_logging


transformers_logging.disable_progress_bar()

VIDEO_CHUNK_SECONDS = 30
VIDEO_BATCH_SIZE = 8
REPEATED_PHRASE_MAX_WORDS = 12
REPEATED_PHRASE_MIN_OCCURRENCES = 3
BART_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
BART_INPUT_WORDS_PER_CHUNK = 900
BART_COPY_NGRAM_BLOCK = 4
BOILERPLATE_LINE_PATTERNS = (
    r"\bback\s+to\s+mail\s+online\s+home\b",
    r"\bback\s+to\s+the\s+page\s+you\s+came\s+from\b",
    r"^back\s+(to|home|from)\b",
    r"^back\s+to\s+.*\s+home$",
    r"^back\s+to\s+the\s+page\s+you\s+came\s+from$",
    r"^mail\s+online\s+home$",
    r"^share\s+this\s+article$",
    r"^read\s+more\b",
    r"^sign\s+in$",
    r"^log\s+in$",
    r"^subscribe\b",
    r"^advertisement$",
    r"^updated:\s*\d",
    r"^published:\s*\d",
    r"^by\s+[a-z ,.'-]+for\s+mailonline$",
    r"\bcybersecurity\s+is\s+cnn\s+tech's\s+weekly\b",
    r"\bfor\s+confidential\s+support\s+call\s+the\s+samaritans\b.*",
    r"\bvisit\s+a\s+local\s+samaritans\s+branch\b.*",
    r"\bclick\s+here\s+for\s+details\b",
)
SUMMARY_LENGTH_OPTIONS = {
    "Short": {
        "words": 70,
        "description": "Best for quick revision notes.",
    },
    "Medium": {
        "words": 130,
        "description": "Balanced detail for assignments and reports.",
    },
    "Long": {
        "words": 220,
        "description": "More complete coverage for longer documents.",
    },
}
COMMON_TRANSCRIPT_FILLERS = (
    "thank you very much for watching this video",
    "thanks for watching",
    "i'll see you in the next video",
    "see you in the next video",
)


# Load the model once
@st.cache_resource
def load_model():
    tokenizer = BartTokenizer.from_pretrained(BART_MODEL_NAME)
    model = BartForConditionalGeneration.from_pretrained(BART_MODEL_NAME)
    model.eval()
    return tokenizer, model


@st.cache_resource
def load_asr_model():
    transformers_logging.disable_progress_bar()
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        device=device,
    )


@st.cache_resource
def load_transcript_api():
    return YouTubeTranscriptApi()

def extract_pdf(file):
    text = ""
    pdf = fitz.open(stream=file.read(), filetype="pdf")

    for page in pdf:
        text += page.get_text()

    return text


def extract_docx(file):
    doc = Document(file)

    text = ""

    for para in doc.paragraphs:
        text += para.text + "\n"

    return text


def extract_txt(file):
    return file.read().decode("utf-8")


class _WebTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._capture_tags = {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "article", "section"}
        self._parts = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "nav", "footer", "header", "aside", "form", "button"}:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "nav", "footer", "header", "aside", "form", "button"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth > 0:
            return
        text = " ".join(data.split())
        if text:
            self._parts.append(text)

    def get_text(self):
        return " ".join(self._parts)


def extract_url_text(url):
    parsed_url = urlparse(url.strip())
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        return "", "Please enter a full http or https URL."

    if is_youtube_url(url):
        return extract_youtube_transcript(url)

    request = Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; AITextSummarizer/1.0)"}
    )

    try:
        with urlopen(request, timeout=20) as response:
            content_type = response.headers.get_content_type()
            if content_type not in {"text/html", "application/xhtml+xml"}:
                return "", "That URL does not look like a readable webpage. Try a news article or blog page."

            html_content = response.read().decode(response.headers.get_content_charset() or "utf-8", errors="ignore")
    except (URLError, HTTPError, ValueError):
        return "", "I could not open that webpage. It may be private, blocked, or unavailable."

    parser = _WebTextExtractor()
    parser.feed(html_content)
    extracted_text = parser.get_text()
    text = " ".join(extracted_text.split())
    if not text:
        return "", "I opened the webpage, but no readable article text was found."
    return text, ""


def is_youtube_url(url):
    parsed_url = urlparse(url.strip())
    host = parsed_url.netloc.lower()
    return "youtube.com" in host or "youtu.be" in host


def extract_youtube_video_id(url):
    parsed_url = urlparse(url.strip())
    if "youtu.be" in parsed_url.netloc.lower():
        return parsed_url.path.lstrip("/").split("/")[0]

    if "youtube.com" in parsed_url.netloc.lower():
        query_params = dict(part.split("=", 1) for part in parsed_url.query.split("&") if "=" in part)
        if "v" in query_params:
            return query_params["v"]

        path_parts = [part for part in parsed_url.path.split("/") if part]
        if len(path_parts) >= 2 and path_parts[0] in {"shorts", "embed", "live"}:
            return path_parts[1]

    return ""


def extract_youtube_transcript(url):
    video_id = extract_youtube_video_id(url)
    if not video_id:
        return "", "I could not recognize the YouTube video ID from that link."

    try:
        transcript_api = load_transcript_api()
        transcript_list = transcript_api.list(video_id)

        transcript = None
        for language_codes in (["en", "en-US", "en-GB"], ["en"]):
            try:
                transcript = transcript_list.find_manually_created_transcript(language_codes)
                break
            except Exception:
                try:
                    transcript = transcript_list.find_generated_transcript(language_codes)
                    break
                except Exception:
                    continue

        if transcript is None:
            transcript = transcript_list.find_transcript(["en", "en-US", "en-GB"])

        transcript_segments = transcript.fetch()
    except Exception:
        return "", "No transcript was available for that YouTube video. Try a video with captions enabled."

    transcript_text = clean_transcript_text(" ".join(segment.text for segment in transcript_segments))
    if not transcript_text:
        return "", "The video transcript was empty."
    return transcript_text, ""


def get_video_duration_seconds(video_path):
    ffmpeg_path = get_ffmpeg_exe()
    result = subprocess.run(
        [ffmpeg_path, "-i", video_path],
        capture_output=True,
        text=True
    )
    duration_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    if not duration_match:
        return 0

    hours, minutes, seconds = duration_match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def transcribe_audio_file(audio_path, asr_model):
    with wave.open(audio_path, "rb") as audio_file:
        frame_count = audio_file.getnframes()
        sample_rate = audio_file.getframerate()
        audio_bytes = audio_file.readframes(frame_count)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    if len(audio_array) == 0:
        return ""

    transcript = asr_model(
        {"array": audio_array, "sampling_rate": sample_rate},
        chunk_length_s=VIDEO_CHUNK_SECONDS,
        batch_size=VIDEO_BATCH_SIZE,
        return_timestamps=False,
    )
    if isinstance(transcript, dict):
        return transcript.get("text", "").strip()
    return str(transcript).strip()


def extract_video_audio(video_path):
    audio_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            audio_path = temp_audio.name

        ffmpeg_path = get_ffmpeg_exe()
        command = [
            ffmpeg_path,
            "-y",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            audio_path,
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            return None
        return audio_path
    except Exception:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
        return None


def extract_video_segment_audio(video_path, start_seconds, duration_seconds):
    audio_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            audio_path = temp_audio.name

        ffmpeg_path = get_ffmpeg_exe()
        command = [
            ffmpeg_path,
            "-y",
            "-ss",
            str(max(0, start_seconds)),
            "-t",
            str(duration_seconds),
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            audio_path,
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            return None
        return audio_path
    except Exception:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
        return None


def extract_video_audio_text(file, full_video=True):
    if hasattr(file, "seek"):
        file.seek(0)

    with tempfile.NamedTemporaryFile(delete=False, suffix="." + file.name.split(".")[-1]) as temp_video:
        temp_video.write(file.read())
        video_path = temp_video.name

    try:
        asr_model = load_asr_model()
        if not full_video:
            duration_seconds = get_video_duration_seconds(video_path)
            if duration_seconds <= 0:
                segment_starts = [0]
                segment_seconds = 20
            else:
                segment_seconds = min(20, max(4, int(duration_seconds)))
                usable_duration = max(1, duration_seconds - segment_seconds)
                segment_count = 6 if duration_seconds > 360 else 4
                segment_starts = [
                    round((usable_duration * index) / max(1, segment_count - 1), 2)
                    for index in range(segment_count)
                ]

            transcript_parts = []
            for start in segment_starts:
                audio_path = extract_video_segment_audio(video_path, start, segment_seconds)
                if not audio_path:
                    continue

                try:
                    part = transcribe_audio_file(audio_path, asr_model)
                    if part:
                        transcript_parts.append(part.strip())
                finally:
                    if os.path.exists(audio_path):
                        os.remove(audio_path)

            return clean_transcript_text(" ".join(transcript_parts))

        audio_path = extract_video_audio(video_path)
        if not audio_path:
            return ""

        try:
            return clean_transcript_text(transcribe_audio_file(audio_path, asr_model))
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)
    finally:
        if video_path and os.path.exists(video_path):
            os.remove(video_path)


def remove_repeated_phrases(text):
    clean_text = " ".join(text.split())
    if not clean_text:
        return ""

    words = clean_text.split()
    for phrase_length in range(REPEATED_PHRASE_MAX_WORDS, 2, -1):
        index = 0
        filtered_words = []
        while index < len(words):
            phrase = words[index:index + phrase_length]
            if len(phrase) < phrase_length:
                filtered_words.extend(words[index:])
                break

            repeats = 1
            while (
                index + (repeats + 1) * phrase_length <= len(words)
                and words[index + repeats * phrase_length:index + (repeats + 1) * phrase_length] == phrase
            ):
                repeats += 1

            filtered_words.extend(phrase)
            index += repeats * phrase_length

        words = filtered_words

    sentence_counts = {}
    unique_sentences = []
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", " ".join(words))
        if sentence.strip()
    ]
    for sentence in sentences:
        sentence_key = re.sub(r"[^a-z0-9]+", " ", sentence.lower()).strip()
        if not sentence_key:
            continue

        sentence_counts[sentence_key] = sentence_counts.get(sentence_key, 0) + 1
        if sentence_counts[sentence_key] == 1:
            unique_sentences.append(sentence)

    return " ".join(unique_sentences) if unique_sentences else " ".join(words)


def is_boilerplate_text(text):
    normalized = re.sub(r"[^a-z0-9:'\s-]+", "", text.lower()).strip()
    normalized = " ".join(normalized.split())
    if not normalized:
        return True

    return any(
        re.search(pattern, normalized, flags=re.IGNORECASE)
        for pattern in BOILERPLATE_LINE_PATTERNS
    )


def remove_boilerplate_text(text):
    clean_text = " ".join(text.split())
    if not clean_text:
        return ""

    for pattern in BOILERPLATE_LINE_PATTERNS:
        clean_text = re.sub(
            pattern,
            " ",
            clean_text,
            flags=re.IGNORECASE,
        )

    parts = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|[\r\n]+", clean_text)
        if part.strip()
    ]
    filtered_parts = [part for part in parts if not is_boilerplate_text(part)]
    return " ".join(filtered_parts) if filtered_parts else clean_text


def clean_transcript_text(text):
    clean_text = remove_boilerplate_text(remove_repeated_phrases(text))
    for filler in COMMON_TRANSCRIPT_FILLERS:
        clean_text = re.sub(
            rf"(?:\b{re.escape(filler)}\b[\s,.;:!-]*){{2,}}",
            "",
            clean_text,
            flags=re.IGNORECASE,
        )
    return " ".join(clean_text.split())


def extractive_summary(clean_text, target_words):
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", clean_text)
        if sentence.strip()
    ]
    if len(sentences) <= 1:
        return " ".join(clean_text.split()[:target_words])

    stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
        "have", "in", "is", "it", "its", "of", "on", "or", "that", "the",
        "this", "to", "was", "were", "with"
    }
    words = [
        word
        for word in re.findall(r"[A-Za-z][A-Za-z'-]*", clean_text.lower())
        if word not in stop_words and len(word) > 2
    ]
    frequencies = {}
    for word in words:
        frequencies[word] = frequencies.get(word, 0) + 1

    sentence_scores = []
    for index, sentence in enumerate(sentences):
        sentence_words = re.findall(r"[A-Za-z][A-Za-z'-]*", sentence.lower())
        useful_words = [word for word in sentence_words if word not in stop_words and len(word) > 2]
        if not useful_words:
            continue

        score = sum(frequencies.get(word, 0) for word in useful_words) / len(useful_words)
        if index == 0:
            score *= 1.15
        sentence_scores.append((score, index, sentence))

    if not sentence_scores:
        return " ".join(clean_text.split()[:target_words])

    selected = []
    selected_word_count = 0
    for _, index, sentence in sorted(sentence_scores, reverse=True):
        sentence_word_count = len(sentence.split())
        if selected and selected_word_count + sentence_word_count > target_words:
            continue
        selected.append((index, sentence))
        selected_word_count += sentence_word_count
        if selected_word_count >= target_words * 0.85:
            break

    if not selected:
        selected = [(sentence_scores[0][1], sentence_scores[0][2])]

    summary = " ".join(sentence for _, sentence in sorted(selected))
    return " ".join(summary.split()[:target_words])


def chunk_text_by_words(text, words_per_chunk):
    words = text.split()
    return [
        " ".join(words[index:index + words_per_chunk])
        for index in range(0, len(words), words_per_chunk)
    ]


def target_abstractive_words(input_word_count, target_words):
    if input_word_count <= 35:
        return min(target_words, input_word_count)
    return max(25, min(target_words, int(input_word_count * 0.35)))


def generate_bart_summary(tokenizer, model, text, max_tokens, min_tokens):
    inputs = tokenizer(
        text,
        max_length=1024,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_tokens,
            min_length=min_tokens,
            num_beams=2,
            no_repeat_ngram_size=3,
            encoder_no_repeat_ngram_size=BART_COPY_NGRAM_BLOCK,
            length_penalty=1.2,
            early_stopping=True,
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()


def abstractive_summary(clean_text, target_words):
    tokenizer, model = load_model()
    input_word_count = len(clean_text.split())
    desired_words = target_abstractive_words(input_word_count, target_words)
    chunks = chunk_text_by_words(clean_text, BART_INPUT_WORDS_PER_CHUNK)
    if not chunks:
        return ""

    chunk_target_words = max(25, int(desired_words / len(chunks)))
    max_tokens_per_chunk = max(35, min(180, int(chunk_target_words * 1.35)))
    min_tokens_per_chunk = max(12, min(80, int(chunk_target_words * 0.45)))
    if min_tokens_per_chunk >= max_tokens_per_chunk:
        min_tokens_per_chunk = max(8, max_tokens_per_chunk - 10)

    summaries = [
        generate_bart_summary(
            tokenizer,
            model,
            chunk,
            max_tokens_per_chunk,
            min_tokens_per_chunk,
        )
        for chunk in chunks
    ]

    summary = " ".join(part for part in summaries if part)
    if not summary:
        return ""

    if len(chunks) > 2 or len(summary.split()) > desired_words * 1.35:
        final_max_tokens = max(30, min(220, int(desired_words * 1.25)))
        final_min_tokens = max(10, min(90, int(desired_words * 0.45)))
        if final_min_tokens >= final_max_tokens:
            final_min_tokens = max(8, final_max_tokens - 10)
        summary = generate_bart_summary(
            tokenizer,
            model,
            summary,
            final_max_tokens,
            final_min_tokens,
        )

    return " ".join(summary.split()[:desired_words])


def generate_summary(text, target_words, use_abstractive=False):
    clean_text = clean_transcript_text(text)
    input_word_count = len(clean_text.split())
    if input_word_count == 0:
        return ""

    if use_abstractive:
        summary = abstractive_summary(clean_text, target_words)
        if summary and summary.strip().lower() != clean_text.strip().lower():
            return summary
        return "The model could not create a shorter generated summary for this input. Try adding more text or choosing a lower target word count."

    return abstractive_summary(clean_text, target_words)


def clean_summary_output(summary):
    return remove_boilerplate_text(summary)

st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    :root {
        --page: #f7f8f5;
        --panel: rgba(255, 255, 255, 0.92);
        --panel-strong: #ffffff;
        --ink: #1f2933;
        --muted: #667085;
        --line: rgba(31, 41, 51, 0.12);
        --teal: #0f9f8f;
        --coral: #e76f51;
        --gold: #f4a261;
        --sage: #dce8dd;
        --shadow: 0 20px 52px rgba(31, 41, 51, 0.11);
    }

    .stApp {
        background:
            linear-gradient(135deg, rgba(220, 232, 221, 0.82), transparent 36%),
            linear-gradient(225deg, rgba(244, 162, 97, 0.16), transparent 30%),
            linear-gradient(180deg, #fbfaf7 0%, #f3f6f1 48%, #f8f8f4 100%);
        color: var(--ink);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.4rem;
        padding-bottom: 2.4rem;
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(31, 41, 51, 0.97), rgba(42, 65, 63, 0.96)),
            #1f2933;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: #e5ecf7;
    }

    [data-testid="stSidebar"] input {
        background: linear-gradient(135deg, #fffaf0 0%, #ecfeff 100%) !important;
        color: #1f2933 !important;
        -webkit-text-fill-color: #1f2933 !important;
        border-radius: 12px;
        border: 1px solid rgba(15, 159, 143, 0.34) !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.85);
        font-weight: 750;
    }

    [data-testid="stSidebar"] .stNumberInput input {
        background: linear-gradient(135deg, #fff7ed 0%, #dffbf5 100%) !important;
        color: #1f2933 !important;
        -webkit-text-fill-color: #1f2933 !important;
    }

    [data-testid="stSidebar"] .stNumberInput button {
        background: linear-gradient(135deg, #1f2933, #0f9f8f) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
    }

    [data-testid="stSidebar"] .stNumberInput button svg {
        fill: #ffffff !important;
        color: #ffffff !important;
        stroke: #ffffff !important;
    }

    .hero {
        position: relative;
        overflow: hidden;
        background:
            linear-gradient(120deg, rgba(31, 41, 51, 0.96), rgba(15, 159, 143, 0.82)),
            linear-gradient(45deg, #1f2933, #0f9f8f);
        border: 1px solid rgba(255, 255, 255, 0.16);
        border-radius: 22px;
        padding: 1.7rem 1.8rem;
        box-shadow: var(--shadow);
        margin-bottom: 1rem;
        min-height: 235px;
        isolation: isolate;
    }

    .hero::before {
        content: "";
        position: absolute;
        inset: 0;
        background-image:
            linear-gradient(rgba(255, 255, 255, 0.08) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.08) 1px, transparent 1px);
        background-size: 44px 44px;
        mask-image: linear-gradient(90deg, rgba(0, 0, 0, 0.25), transparent 74%);
        z-index: -1;
    }

    .hero::after {
        content: "";
        position: absolute;
        right: -90px;
        top: -120px;
        width: 340px;
        height: 340px;
        border: 56px solid rgba(255, 255, 255, 0.1);
        border-radius: 50%;
        z-index: -1;
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.32rem 0.7rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.14);
        color: #ecfdf5;
        font-size: 0.83rem;
        font-weight: 700;
        letter-spacing: 0;
        margin-bottom: 0.85rem;
        border: 1px solid rgba(255, 255, 255, 0.16);
    }

    .hero h1 {
        margin: 0;
        color: #ffffff;
        font-size: 2.7rem;
        line-height: 1.04;
        letter-spacing: 0;
        max-width: 780px;
    }

    .hero p {
        margin: 0.65rem 0 0;
        color: #eef7f2;
        font-size: 1.02rem;
        max-width: 64ch;
    }

    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 1rem;
    }

    .chip {
        border: 1px solid rgba(255, 255, 255, 0.16);
        background: rgba(255, 255, 255, 0.12);
        color: #f8fafc;
        border-radius: 999px;
        padding: 0.35rem 0.72rem;
        font-size: 0.82rem;
        font-weight: 650;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.8rem;
        margin: 0 0 1rem;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 0.95rem 1rem;
        box-shadow: 0 12px 32px rgba(20, 33, 61, 0.08);
        backdrop-filter: blur(14px);
    }

    .metric-card span {
        display: block;
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.35rem;
    }

    .metric-card strong {
        display: block;
        color: var(--ink);
        font-size: 1.05rem;
        line-height: 1.25;
    }

    .panel {
        background: var(--panel-strong);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1.05rem 1.05rem 1.15rem;
        box-shadow: 0 16px 42px rgba(20, 33, 61, 0.09);
        backdrop-filter: blur(16px);
    }

    .panel-title {
        font-size: 0.9rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.75rem;
        font-weight: 800;
    }

    .output-box {
        background:
            linear-gradient(135deg, rgba(31, 41, 51, 0.98), rgba(42, 65, 63, 0.98));
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 1rem 1.05rem;
        color: #f8fafc;
        line-height: 1.7;
        white-space: pre-wrap;
        word-wrap: break-word;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
    }

    .summary-box {
        background: linear-gradient(135deg, #0f9f8f, #e76f51);
        border: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 0 16px 34px rgba(20, 33, 61, 0.18);
    }

    .sidebar-card {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.11);
        border-radius: 16px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 12px 26px rgba(0, 0, 0, 0.12);
    }

    .sidebar-card h3 {
        margin: 0 0 0.35rem;
        color: #ffffff;
        font-size: 1rem;
    }

    .sidebar-card p {
        margin: 0;
        color: #d1d9e6;
        font-size: 0.92rem;
        line-height: 1.55;
    }

    .workflow-card {
        background: linear-gradient(180deg, rgba(31, 41, 51, 0.96), rgba(15, 159, 143, 0.86));
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 16px 36px rgba(20, 33, 61, 0.18);
    }

    .workflow-card h3 {
        margin: 0 0 0.35rem;
        color: #ffffff;
        font-size: 1rem;
    }

    .workflow-card p {
        margin: 0;
        color: #e5ecf7;
        font-size: 0.92rem;
        line-height: 1.55;
    }

    .stButton > button {
        width: 100%;
        border-radius: 13px;
        background: linear-gradient(135deg, var(--teal), var(--coral));
        color: #ffffff;
        border: none;
        font-weight: 800;
        padding: 0.78rem 1rem;
        box-shadow: 0 14px 28px rgba(15, 159, 143, 0.2);
        transition: transform 160ms ease, filter 160ms ease, box-shadow 160ms ease;
    }

    .stButton > button:hover {
        filter: brightness(1.03);
        transform: translateY(-1px);
        box-shadow: 0 18px 34px rgba(15, 159, 143, 0.25);
    }

    .stTextArea textarea,
    .stTextInput input,
    .stNumberInput input,
    .stFileUploader section {
        border-radius: 13px !important;
    }

    .stTextArea textarea,
    .stTextInput input {
        background: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #d5dce8 !important;
        box-shadow: 0 10px 24px rgba(20, 33, 61, 0.07);
    }

    .stTextArea textarea:focus,
    .stTextInput input:focus {
        border-color: var(--teal) !important;
        box-shadow: 0 0 0 3px rgba(15, 159, 143, 0.16);
    }

    .stTextArea textarea::placeholder,
    .stTextInput input::placeholder {
        color: #64748b !important;
    }

    .stAlert {
        border-radius: 14px;
    }

    .stAlert [data-testid="stAlertContainer"] {
        color: var(--ink);
    }

    .stRadio [role="radiogroup"] {
        gap: 0.5rem;
    }

    .stRadio label {
        background: #ffffff;
        border: 1px solid var(--line);
        padding: 0.5rem 0.85rem;
        border-radius: 999px;
        color: var(--ink);
        min-width: 128px;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        justify-content: flex-start;
        white-space: nowrap;
        box-shadow: 0 8px 20px rgba(20, 33, 61, 0.06);
    }

    .stRadio label p,
    .stRadio label span,
    .stRadio label div {
        color: var(--ink) !important;
        font-weight: 600;
    }

    .stRadio label:has(input:checked) {
        border-color: rgba(15, 159, 143, 0.38);
        background: #effaf7;
        box-shadow: 0 10px 22px rgba(15, 159, 143, 0.12);
    }

    .output-box strong,
    .output-box em,
    .output-box code,
    .summary-box strong,
    .summary-box em,
    .summary-box code {
        color: #ffffff;
    }

    .stSelectbox div[data-baseweb="select"] > div {
        border-radius: 13px;
        border-color: var(--line);
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #ffffff;
        border: 1px dashed rgba(15, 159, 143, 0.36);
    }

    [data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(135deg, #f7fff9 0%, #ecfeff 100%) !important;
        color: #1f2933 !important;
        border: 1px solid rgba(15, 159, 143, 0.36) !important;
        box-shadow: 0 8px 18px rgba(31, 41, 51, 0.08) !important;
    }

    [data-testid="stFileUploaderDropzone"] button:hover {
        background: linear-gradient(135deg, #effaf7 0%, #fff7ed 100%) !important;
        color: #1f2933 !important;
        border-color: rgba(231, 111, 81, 0.42) !important;
        transform: none;
    }

    [data-testid="stFileUploaderDropzone"] button *,
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] svg {
        color: #1f2933 !important;
        fill: #1f2933 !important;
        stroke: #1f2933 !important;
    }

    @media (max-width: 760px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero {
            padding: 1.25rem;
            min-height: auto;
        }

        .hero h1 {
            font-size: 2rem;
        }

        .metric-grid {
            grid-template-columns: 1fr;
        }

        .stRadio [role="radiogroup"] {
            flex-direction: column;
        }

        .stRadio label {
            width: 100%;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">AI DOCUMENT + VIDEO SUMMARIZER</div>
        <h1>AI Text Summarization</h1>
        <p>Summarize pasted text, documents, or narrated videos with a cleaner workflow and a professional dashboard-style layout.</p>
        <div class="chip-row">
            <div class="chip">PDF, DOCX, TXT</div>
            <div class="chip">Video transcription</div>
            <div class="chip">Word-target summary control</div>
            <div class="chip">Whisper + BART</div>
        </div>
    </div>
    <div class="metric-grid">
        <div class="metric-card">
            <span>Input Modes</span>
            <strong>Text, URL, files, and videos</strong>
        </div>
        <div class="metric-card">
            <span>Output Control</span>
            <strong>Adjustable summary length</strong>
        </div>
        <div class="metric-card">
            <span>Processing</span>
            <strong>Fast previews before summarizing</strong>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

sidebar = st.sidebar
sidebar.markdown(
    """
    <div class="sidebar-card">
        <h3>Summary Length</h3>
        <p>Choose how detailed the generated summary should be.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

summary_length = sidebar.select_slider(
    "Summary Length",
    options=list(SUMMARY_LENGTH_OPTIONS.keys()),
    value="Medium",
)
summary_target_words = SUMMARY_LENGTH_OPTIONS[summary_length]["words"]
sidebar.markdown(
    f"""
    <div class="sidebar-card">
        <h3>Selected Length</h3>
        <p><strong>{summary_length}</strong> summary. {SUMMARY_LENGTH_OPTIONS[summary_length]["description"]}</p>
    </div>
    <div class="sidebar-card">
        <h3>Supported Inputs</h3>
        <p>Paste text, upload PDF/DOCX/TXT, or upload a narrated MP4/MOV/AVI/MKV video.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="panel" style="margin-bottom:1rem;">', unsafe_allow_html=True)
st.markdown('<div class="panel-title">Choose Input Method</div>', unsafe_allow_html=True)
st.markdown('<div style="margin:0.25rem 0 0.55rem; color: var(--muted); font-size: 0.92rem;">Pick one input source to start summarizing.</div>', unsafe_allow_html=True)
option = st.radio(
    "Choose Input Method",
    ["Paste Text", "Paste URL", "Upload File", "Upload Video"],
    horizontal=True,
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

text = ""
video_file = None
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "url_input" not in st.session_state:
    st.session_state.url_input = ""
if "url_warning" not in st.session_state:
    st.session_state.url_warning = ""
if "fetched_url" not in st.session_state:
    st.session_state.fetched_url = ""
if "video_signature" not in st.session_state:
    st.session_state.video_signature = ""
if "video_transcript" not in st.session_state:
    st.session_state.video_transcript = ""
if st.session_state.get("active_option") != option:
    st.session_state.input_text = ""
    st.session_state.url_warning = ""
    st.session_state.fetched_url = ""
    st.session_state.video_signature = ""
    st.session_state.video_transcript = ""
    st.session_state.active_option = option

st.markdown('<div class="panel">', unsafe_allow_html=True)

if option == "Paste Text":
    st.markdown('<div class="panel-title">Text Input</div>', unsafe_allow_html=True)
    text = st.text_area(
        "Paste your text",
        height=300,
        placeholder="Paste a long article, report, notes, or paragraph here...",
        label_visibility="collapsed"
    )
    st.session_state.input_text = text
    if text.strip():
        st.markdown('<div class="panel-title" style="margin-top:1rem;">Live Preview</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="output-box">{html.escape(text[:3000])}</div>', unsafe_allow_html=True)

elif option == "Paste URL":
    st.markdown('<div class="panel-title">Webpage or YouTube URL</div>', unsafe_allow_html=True)
    page_url = st.text_input(
        "Enter a webpage or YouTube URL",
        placeholder="https://www.youtube.com/watch?v=..."
    )
    current_url = page_url.strip()
    if current_url != st.session_state.url_input:
        st.session_state.input_text = ""
        st.session_state.url_warning = ""
        st.session_state.fetched_url = ""
    st.session_state.url_input = current_url

    if current_url:
        st.info("Click Generate Summary to fetch this URL and summarize it in one step.")

elif option == "Upload File":
    st.markdown('<div class="panel-title">Document Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX or TXT",
        type=["pdf", "docx", "txt"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".pdf"):
            text = extract_pdf(uploaded_file)
        elif uploaded_file.name.endswith(".docx"):
            text = extract_docx(uploaded_file)
        elif uploaded_file.name.endswith(".txt"):
            text = extract_txt(uploaded_file)

        st.session_state.input_text = text

        st.markdown('<div class="panel-title" style="margin-top:1rem;">Extracted Text</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="output-box">{html.escape(text[:3000])}</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="panel-title">Video Upload</div>', unsafe_allow_html=True)
    video_file = st.file_uploader(
        "Upload MP4, MOV, AVI or MKV",
        type=["mp4", "mov", "avi", "mkv"],
        label_visibility="collapsed"
    )
    st.caption("Video summaries use a fast transcript preview so large uploads finish sooner.")

    if video_file is not None:
        video_signature = f"{video_file.name}:{getattr(video_file, 'size', 0)}:fast"
        if st.session_state.video_signature == video_signature and st.session_state.video_transcript:
            text = st.session_state.video_transcript
            st.session_state.input_text = text
        else:
            st.session_state.input_text = ""

        if text.strip():
            st.markdown('<div class="panel-title" style="margin-top:1rem;">Transcript Preview</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="output-box">{html.escape(text[:3000])}</div>', unsafe_allow_html=True)
        else:
            st.info("Click Generate Summary to transcribe and summarize this video.")

st.markdown('</div>', unsafe_allow_html=True)

text = st.session_state.get("input_text", "")

action_left, action_right = st.columns([2.7, 1])
with action_left:
    generate_clicked = st.button("Generate Summary", use_container_width=True)
with action_right:
    st.markdown(
        """
        <div class="workflow-card">
            <h3>Workflow</h3>
            <p>1. Choose an input type.<br/>2. Add text, document, or video.<br/>3. Generate a summary.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if generate_clicked:
    if (
        option == "Paste URL"
        and st.session_state.url_input
        and (
            text.strip() == ""
            or st.session_state.fetched_url != st.session_state.url_input
        )
    ):
        with st.spinner("Fetching URL content..."):
            text, fetch_error = extract_url_text(st.session_state.url_input)
            st.session_state.input_text = text
            st.session_state.url_warning = fetch_error
            st.session_state.fetched_url = st.session_state.url_input if text.strip() else ""

        if fetch_error:
            text = ""
            st.session_state.input_text = ""

        if text.strip() != "":
            st.markdown('<div class="panel-title" style="margin-top:1rem;">Fetched Text Preview</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="output-box">{html.escape(text[:3000])}</div>', unsafe_allow_html=True)

    if option == "Upload Video" and text.strip() == "" and video_file is not None:
        video_signature = f"{video_file.name}:{getattr(video_file, 'size', 0)}:fast"

        if st.session_state.video_signature == video_signature and st.session_state.video_transcript:
            text = st.session_state.video_transcript
            st.session_state.input_text = text
        else:
            with st.spinner("Transcribing a fast video summary..."):
                text = extract_video_audio_text(video_file, full_video=False)
                st.session_state.video_signature = video_signature
                st.session_state.video_transcript = text
                st.session_state.input_text = text

        if text.strip():
            st.markdown('<div class="panel-title" style="margin-top:1rem;">Transcript Preview</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="output-box">{html.escape(text[:3000])}</div>', unsafe_allow_html=True)

    if text.strip() == "":
        if option == "Paste URL":
            st.warning(st.session_state.url_warning or "Please enter a valid webpage or YouTube URL.")
        elif option == "Upload Video":
            st.warning("Please upload a video with detectable speech.")
        else:
            st.warning("Please enter some text or upload a supported file.")
    else:
        text = clean_transcript_text(text)
        st.session_state.input_text = text

        with st.spinner("Generating summary..."):
            summary = generate_summary(
                text,
                summary_target_words,
                use_abstractive=True,
            )
            summary = clean_summary_output(summary)

        st.markdown('<div class="panel-title" style="margin-top:1.2rem;">Generated Summary</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="output-box summary-box">{html.escape(summary)}</div>', unsafe_allow_html=True)
