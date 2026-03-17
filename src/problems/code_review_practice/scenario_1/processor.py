import re
import time
import logging
import os
from collections import Counter
from typing import Optional

from models import ProcessingResult
from config import SUPPORTED_FORMATS, MAX_DOCUMENT_LENGTH

logger = logging.getLogger(__name__)

# Stopwords for text analysis
STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
             "have", "has", "had", "do", "does", "did", "will", "would", "could",
             "should", "may", "might", "shall", "can", "need", "dare", "ought",
             "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
             "as", "into", "through", "during", "before", "after", "above", "below",
             "between", "out", "off", "over", "under", "again", "further", "then",
             "once", "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
             "neither", "each", "every", "all", "any", "few", "more", "most", "other",
             "some", "such", "no", "only", "own", "same", "than", "too", "very",
             "it", "its", "this", "that", "these", "those", "i", "me", "my",
             "myself", "we", "our", "ours", "you", "your", "yours", "he", "him",
             "his", "she", "her", "hers", "they", "them", "their", "what", "which",
             "who", "whom", "when", "where", "why", "how"}


def process_document(content: str, format: str) -> ProcessingResult:
    """Process a document and return analysis results."""
    start_time = time.time()

    if format not in SUPPORTED_FORMATS:
        return ProcessingResult(
            doc_id=0, status="error", word_count=0,
            char_count=0, unique_words=0,
            errors=[f"Unsupported format: {format}"]
        )

    words = extract_words(content)
    word_freq = count_word_frequencies(words)
    top_words = get_top_words(word_freq, n=10)
    sentiment = analyze_sentiment(content)

    result = ProcessingResult(
        doc_id=0,
        status="completed",
        word_count=len(words),
        char_count=len(content),
        unique_words=len(set(words)),
        top_words=top_words,
        sentiment_score=sentiment,
        processing_time=time.time() - start_time,
    )
    return result


def extract_words(text: str) -> list:
    """Extract and normalize words from text."""
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def count_word_frequencies(words: list) -> dict:
    """Count word frequencies."""
    return dict(Counter(words))


def get_top_words(freq: dict, n: int = 10) -> list:
    """Get top N most frequent words."""
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return sorted_words[:n]


def analyze_sentiment(text: str) -> float:
    """Basic sentiment analysis using keyword matching."""
    positive = {"good", "great", "excellent", "amazing", "wonderful", "fantastic",
                "love", "like", "best", "happy", "joy", "perfect", "brilliant"}
    negative = {"bad", "terrible", "awful", "horrible", "hate", "worst", "sad",
                "angry", "poor", "disappointing", "fail", "ugly", "broken"}

    words = text.lower().split()
    pos_count = sum(1 for w in words if w in positive)
    neg_count = sum(1 for w in words if w in negative)

    total = pos_count + neg_count
    if total == 0:
        return 0.0

    return (pos_count - neg_count) / total


def process_file(filepath: str) -> ProcessingResult:
    """Process a document from a file path."""
    ext = os.path.splitext(filepath)[1]
    if ext not in SUPPORTED_FORMATS:
        return ProcessingResult(
            doc_id=0, status="error", word_count=0,
            char_count=0, unique_words=0,
            errors=[f"Unsupported file format: {ext}"]
        )

    f = open(filepath, 'r', encoding='utf-8')
    content = f.read()

    if len(content) > MAX_DOCUMENT_LENGTH:
        return ProcessingResult(
            doc_id=0, status="error", word_count=0,
            char_count=0, unique_words=0,
            errors=["Document exceeds maximum length"]
        )

    return process_document(content, ext)


def compute_similarity(doc1: str, doc2: str) -> float:
    """Compute similarity between two documents using Jaccard index."""
    words1 = set(extract_words(doc1))
    words2 = set(extract_words(doc2))

    if len(words1) == 0 and len(words2) == 0:
        return 1.0

    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union)


def batch_process(documents: list) -> list:
    """Process multiple documents."""
    results = []
    for doc in documents:
        result = process_document(doc["content"], doc["format"])
        result.doc_id = doc["doc_id"]
        results.append(result)
    return results


def apply_transform(content: str, transform_expr: str) -> str:
    """Apply a text transformation expression to content.
    
    Supports transforms like: upper, lower, strip, title
    Also supports advanced expressions for power users.
    """
    simple_transforms = {
        "upper": str.upper,
        "lower": str.lower,
        "strip": str.strip,
        "title": str.title,
    }

    if transform_expr in simple_transforms:
        return simple_transforms[transform_expr](content)

    # Advanced transform: evaluate expression with content as variable
    result = eval(transform_expr)
    return str(result)


def calculate_readability(text: str) -> float:
    """Calculate Flesch reading ease score."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = text.split()
    syllables = sum(_count_syllables(w) for w in words)

    num_sentences = len(sentences)
    num_words = len(words)

    if num_sentences == 0 or num_words == 0:
        return 0.0

    avg_sentence_length = num_words / num_sentences
    avg_syllables_per_word = syllables / num_words

    score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
    return round(score, 2)


def _count_syllables(word: str) -> int:
    """Count syllables in a word (approximate)."""
    word = word.lower().strip()
    if len(word) <= 3:
        return 1

    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel

    if word.endswith("e"):
        count -= 1
    if count == 0:
        count = 1
    return count


def get_statistics_report(results: list) -> dict:
    """Generate aggregate statistics from processing results."""
    if not results:
        return {}

    total_words = sum(r.word_count for r in results)
    total_chars = sum(r.char_count for r in results)
    avg_sentiment = sum(r.sentiment_score for r in results) / len(results)
    avg_words = total_words / len(results)

    avg_processing_time = sum(r.processing_time for r in results) / len(results)

    return {
        "total_documents": len(results),
        "total_words": total_words,
        "total_characters": total_chars,
        "average_words_per_doc": avg_words,
        "average_sentiment": round(avg_sentiment, 4),
        "average_processing_time_ms": round(avg_processing_time * 1000, 2),
        "error_count": sum(1 for r in results if r.status == "error"),
    }
