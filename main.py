import nltk
import random
from nltk.corpus import shakespeare
from nltk.tokenize import RegexpTokenizer
from xml.etree import ElementTree
from collections import defaultdict
from nltk.util import ngrams

#Downloading the text
nltk.download('shakespeare')

#Extracting the play according to the nltk official site
play_name = 'hamlet.xml'  
play = shakespeare.xml(play_name)

#Extracting spoken text from the play
def extract_text(play):
    lines = [line.text for line in play.findall('.//LINE') if line.text]
    return " ".join(lines)

text = extract_text(play)

#Lowercase and no punctuation using RegexpTokenizer
def preprocess_text(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text.lower())
    return tokens

#Printing the first 50 tokens as an example
tokens = preprocess_text(text)
print("\nSample:", tokens[:50])  

#Generating ngram counts. It is an ngram function so the model can be used for tri-grams and quad-grams as well
def generate_ngram_counts(tokens, n):
    ngram_counts = defaultdict(lambda: defaultdict(int))
    
    for ngram in ngrams(tokens, n):
        prefix, next_word = tuple(ngram[:-1]), ngram[-1]
        ngram_counts[prefix][next_word] += 1
    
    return ngram_counts

#Computing probabilities of next token given the prefix
def compute_probabilities(ngram_counts):
    ngram_probs = {}
    for prefix, word_counts in ngram_counts.items():
        total_count = sum(word_counts.values())
        word_probs = {word: count / total_count for word, count in word_counts.items()}
        ngram_probs[prefix] = word_probs
    
    return ngram_probs

#Generate next word(token)
def get_next_token(prefix, ngram_probs):
    if prefix in ngram_probs:
        next_words, probabilities = zip(*ngram_probs[prefix].items())
        return random.choices(next_words, probabilities)[0]
    else:
        # Fallback: Choose a high-frequency n-gram start instead of a random word
        return random.choice(list(ngram_probs.keys()))[0]


#Selecting a frequent ngram to generate better sentences
def get_frequent_start_ngram(ngram_counts):
    if not ngram_counts:
        return None
    
    common_ngrams = sorted(ngram_counts.keys(), key=lambda k: sum(ngram_counts[k].values()), reverse=True)
    return random.choice(common_ngrams[:50])

#Generating text
def generate_text(initial_ngram, ngram_probs, num_words):
    if not initial_ngram:
        return "Error: No valid starting n-gram found."
    
    current_ngram = initial_ngram
    generated_text = list(initial_ngram)
    
    for _ in range(num_words - len(initial_ngram)):
        next_word = get_next_token(current_ngram, ngram_probs)
        if not next_word:
            break
        
        generated_text.append(next_word)
        current_ngram = tuple(generated_text[-(len(initial_ngram)):])  # Shift n-gram window
    
    return ' '.join(generated_text)


#Generating for bigram, trigram and quadgram
bigram_counts = generate_ngram_counts(tokens, 2)
bigram_probs = compute_probabilities(bigram_counts)

trigram_counts = generate_ngram_counts(tokens, 3)
trigram_probs = compute_probabilities(trigram_counts)

quadgram_counts = generate_ngram_counts(tokens, 4)
quadgram_probs = compute_probabilities(quadgram_counts)

#Example of generated text
print("\nGenerated Text (Bigram):", generate_text(('to', 'be'), bigram_probs, 50))
print("\nGenerated Text (Trigram):", generate_text(('to', 'be', 'or'), trigram_probs, 50))
print("\nGenerated Text (Quadgram):", generate_text(('to', 'be', 'or', 'not'), quadgram_probs, 50))
