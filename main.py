import nltk
import random
from nltk.corpus import shakespeare
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from xml.etree import ElementTree
from collections import defaultdict
from nltk.util import ngrams

#Downloading the text
nltk.download('shakespeare')
nltk.download('punkt')
nltk.download('punkt_tab')


#Extracting the play according to the nltk official site
play_name = 'hamlet.xml'  
play = shakespeare.xml(play_name)

#Extracting spoken text from the play
def extract_text(play):
    lines = [line.text for line in play.findall('.//LINE') if line.text]
    return " ".join(lines)

text = extract_text(play)

sentences = sent_tokenize(text)

#Lowercase and no punctuation using RegexpTokenizer
def preprocess_text(sentence):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence.lower())
    return tokens

#Printing the first 50 tokens as an example
tokens = preprocess_text(text)
print("\nSample:", tokens[:50])  

#Generating ngram counts. It is an ngram function so the model can be used for tri-grams and quad-grams as well
def generate_ngram_counts(sentences, n):
    ngram_counts = defaultdict(lambda: defaultdict(int))
    sentence_starts = []  # Store the first (n-1)-grams of sentences

    for sentence in sentences:
        tokens = preprocess_text(sentence)  # Ensure tokenization
        
        if len(tokens) < n:  # Skip short sentences
            continue

        first_ngram = tuple(tokens[:n-1])  # Store first (n-1)-gram as sentence starter
        sentence_starts.append(first_ngram)

        # Generate proper n-grams
        for i in range(len(tokens) - (n - 1)):  
            prefix = tuple(tokens[i:i + n - 1])  # Extract (n-1) prefix
            next_word = tokens[i + n - 1]  # Get next word
            ngram_counts[prefix][next_word] += 1  # Store n-gram correctly

    return ngram_counts, sentence_starts



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
    return None  #No next word available

#Selecting a random sentence-starting n-gram
def get_random_sentence_start(sentence_starts, ngram_counts):
    if sentence_starts:
        return random.choice(sentence_starts) 
    elif ngram_counts:
        return random.choice(list(ngram_counts.keys()))  #Fallback to a random valid n-gram
    return None  

#Generating text ensuring complete sentences
def generate_sentence(initial_ngram, ngram_probs, max_words=50):
    if not initial_ngram:
        return "Error: No valid starting n-gram found."

    current_ngram = initial_ngram
    generated_text = list(initial_ngram)

    for _ in range(max_words - len(initial_ngram)):
        next_word = get_next_token(current_ngram, ngram_probs)
        if not next_word:
            generated_text.append(".")  # Ensure sentence ends properly
            break

        generated_text.append(next_word)
        current_ngram = tuple(generated_text[-(len(initial_ngram)):])  # Shift n-gram window

    generated_text[0] = generated_text[0].capitalize()  # Capitalize first word
    if not generated_text[-1].endswith("."):
        generated_text.append(".")  # Add a full stop if it’s missing
    
    return ' '.join(generated_text)

    
    # Ensure the first word is capitalized
    generated_text[0] = generated_text[0].capitalize()
    return ' '.join(generated_text)


# Generating n-grams per sentence
bigram_counts, bigram_starts = generate_ngram_counts(sentences, 2)
bigram_probs = compute_probabilities(bigram_counts)

trigram_counts, trigram_starts = generate_ngram_counts(sentences, 3)
trigram_probs = compute_probabilities(trigram_counts)

quadgram_counts, quadgram_starts = generate_ngram_counts(sentences, 4)
quadgram_probs = compute_probabilities(quadgram_counts)

# Generate and print 5 sentences for each n-gram model
print("\nGenerated Sentences (Bigram):")
for _ in range(5):
    initial_bigram = get_random_sentence_start(bigram_starts, bigram_counts)
    print(generate_sentence(initial_bigram, bigram_probs))

print("\nGenerated Sentences (Trigram):")
for _ in range(5):
    initial_trigram = get_random_sentence_start(trigram_starts, trigram_counts)
    print(generate_sentence(initial_trigram, trigram_probs))

print("\nGenerated Sentences (Quadgram):")
for _ in range(5):
    initial_quadgram = get_random_sentence_start(quadgram_starts, quadgram_counts)
    print(generate_sentence(initial_quadgram, quadgram_probs))
