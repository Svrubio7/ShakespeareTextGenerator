import pytest
import random
from collections import defaultdict
from main import (
    preprocess_text,
    generate_ngram_counts,
    compute_probabilities,
    get_random_sentence_start,
    get_next_token,
    generate_sentence
)

@pytest.fixture
def sample_data():
    sentences = [
        "To be or not to be.",
        "The lady doth protest too much, methinks.",
        "All the world’s a stage.",
        "Brevity is the soul of wit."
    ]
    processed_sentences = [preprocess_text(sent) for sent in sentences]
    
    bigram_counts, bigram_starts = generate_ngram_counts(sentences, 2)
    bigram_probs = compute_probabilities(bigram_counts)
    return sentences, processed_sentences, bigram_counts, bigram_starts, bigram_probs

# Testing if text preprocessing removes punctuation and lowercases text
def test_preprocess_text():
    result = preprocess_text("To be, or not to be!")
    assert result == ["to", "be", "or", "not", "to", "be"]

# Testing bigram generation
def test_generate_bigram_counts(sample_data):
    _, _, bigram_counts, bigram_starts, _ = sample_data
    assert ('to', 'be') in bigram_counts, f"Expected ('to', 'be') in bigram_counts, but got {list(bigram_counts.keys())}"
    assert ('be', 'or') in bigram_counts, "Expected ('be', 'or') in bigram_counts"
    assert len(bigram_counts) > 0
    assert ('to',) in bigram_starts, "Expected ('to',) in bigram_starts"

# Testing trigram generation
def test_generate_trigram_counts():
    sentences = [
        "To be or not to be.",
        "The lady doth protest too much, methinks."
    ]
    trigram_counts, trigram_starts = generate_ngram_counts(sentences, 3)
    assert ('to', 'be', 'or') in trigram_counts, "Expected ('to', 'be', 'or') in trigram_counts"
    assert ('be', 'or', 'not') in trigram_counts, "Expected ('be', 'or', 'not') in trigram_counts"
    assert ('to', 'be') in trigram_starts, "Expected ('to', 'be') in trigram_starts"

# Testing quadgram generation
def test_generate_quadgram_counts():
    sentences = [
        "To be or not to be.",
        "The lady doth protest too much, methinks."
    ]
    quadgram_counts, quadgram_starts = generate_ngram_counts(sentences, 4)
    assert ('to', 'be', 'or', 'not') in quadgram_counts, "Expected ('to', 'be', 'or', 'not') in quadgram_counts"
    assert ('to', 'be', 'or') in quadgram_starts, "Expected ('to', 'be', 'or') in quadgram_starts"

# Testing if probabilities sum to 1 for each prefix
def test_compute_probabilities(sample_data):
    _, _, _, _, bigram_probs = sample_data
    for prefix, probs in bigram_probs.items():
        assert pytest.approx(sum(probs.values()), 0.00001) == 1

# Testing if a valid sentence start is chosen
def test_get_random_sentence_start(sample_data):
    _, _, bigram_counts, bigram_starts, _ = sample_data
    start = get_random_sentence_start(bigram_starts, bigram_counts)
    assert start in bigram_counts

# Testing if a next token is chosen correctly based on probabilities
def test_get_next_token(sample_data):
    _, _, _, _, bigram_probs = sample_data
    prefix = ('to', 'be')
    if prefix in bigram_probs:
        next_token = get_next_token(prefix, bigram_probs)
        assert next_token in bigram_probs[prefix]

# Testing if generated sentences start with a capital letter and end with a full stop
def test_generate_sentence(sample_data):
    _, _, bigram_counts, bigram_starts, bigram_probs = sample_data
    start = get_random_sentence_start(bigram_starts, bigram_counts)
    sentence = generate_sentence(start, bigram_probs, max_words=10)
    assert sentence[0].isupper()
    assert sentence.endswith('.')