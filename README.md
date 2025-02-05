# Shakespeare Text Generator

## Overview
This project is a Natural Language Processing (NLP) implementation that generates text in the style of Shakespeare. It utilizes **n-gram language modeling** to predict and generate sequences based on Shakespeare's plays.

## Features
- Tokenization and preprocessing of Shakespearean text
- **n-gram** models (bigrams, trigrams, quadgrams)
- Probabilistic next-word prediction
- Sentence generation
- Unit tests for validation

## Installation
### Prerequisites
Ensure you have **Python 3.10+** installed.

### Required Libraries
Run the following command to install dependencies:

```bash
pip install -r requirements.txt
```

## Usage
### Running the Script
To generate sentences using Shakespearean text:

```bash
python main.py
```

## Code Structure
```
ShakespeareTextGenerator/
│── main.py            # Core logic for text generation
│── testing.py         # Unit tests for validation
│── README.md          # Documentation
│── requirements.txt   # List of dependencies
```

### Main Components
#### 1. **Text Preprocessing**
- Uses `nltk` for tokenization.
- Removes punctuation and converts text to lowercase.

#### 2. **n-gram Model Generation**
- The function `generate_ngram_counts(sentences, n)` constructs **bigram, trigram, and quadgram** models.
- Uses `defaultdict` for efficient counting.
- Stores **(n-1)-gram prefixes** and their corresponding next-word probabilities.

#### 3. **Text Generation**
- `generate_sentence(initial_ngram, ngram_probs, max_words=50)` generates sentences probabilistically.
- Sentences now strictly follow the trained n-gram model structure.

## Recent Modifications
### 🔹 **Refactored `generate_ngram_counts`**
- **Bug Fix:** Ensured proper tuple formation for n-grams.
- **Debugging:** Added controlled logging to verify stored n-grams.
- **Impact:** Now correctly tracks `n-1` prefixes but requires improved handling for starting sequences.

### 🔹 **Unit Test Adjustments**
- Updated `test_generate_ngram_counts` to check correct tuple storage.
- Fixed `test_get_random_sentence_start` to align with n-gram tuple structure.
- Debugging information now logs a **single stored n-gram** for clarity.

## Running Tests
To validate functionality:
```bash
pytest testing.py -s
```
This runs all unit tests and provides debugging output for **n-gram storage.**

## Known Issues
- Sentences sometimes terminate early (**n-1 words only**).
- `test_get_random_sentence_start` intermittently fails due to prefix mismatch.

## Future Improvements
- Improve handling of `n-1` prefixes for better sentence initialization.
- Refactor `generate_sentence` to ensure sentences reach full length.
- Implement **smoother probabilistic transitions** for better coherence.


## Author
Sergio Verdugo - [Svrubio7](https://github.com/Svrubio7)
