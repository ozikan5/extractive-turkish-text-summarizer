import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

TURKISH_STOPWORDS = ['bir', 'ile', 've', 'de', 'ama', 'bu', 'çok', 'da', 'gibi', 'şu']


# Split the sentences in the text 
def sentence_tokenizer(text):
    return re.split(r'(?<=[.!?])\s+', text.strip())

# Split words in each sentence
def turkish_word_tokenizer(text):
    text = re.sub(r'\s+', ' ', text.strip())
    text = text.lower().replace("i̇", "i")
    tokens = re.findall(r"\b[\wçğıöşüÇĞİÖŞÜ]+(?:'\w+)?\b", text, flags=re.UNICODE)
    return tokens

# Build the frequency table for each word to assess importance
def build_frequency_table_from_tokens(tokenized_sentences):
    freq_table = defaultdict(int)
    for tokens in tokenized_sentences:
        for word in tokens:
            freq_table[word] += 1
    return freq_table

# Score each sentence based on frequency table
def score_sentences_from_tokens(sentences, tokenized_sentences, freq_table):
    sentence_scores = {}
    for sent, tokens in zip(sentences, tokenized_sentences):
        score = sum(freq_table[word] for word in tokens)
        sentence_scores[sent] = score / len(tokens) if tokens else 0
    return sentence_scores

# Pick custom number of most important sentences to summarize
def get_summary(sentences, sentence_scores, top_n=2):
    ranked = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
    selected = [s for s, _ in ranked[:top_n]]
    return ' '.join(sorted(selected, key=lambda s: sentences.index(s)))

# Open custom text file to summarize
filename = str(input("The text you want to summarize: "))
file = open(filename)
text = file.read()

# Tokenize sentences and remove unnecessary connection words for clarity
sentences = sentence_tokenizer(text)

vectorizer = TfidfVectorizer(
    tokenizer=turkish_word_tokenizer,
    stop_words=TURKISH_STOPWORDS,
    lowercase=True
)

tfidf_matrix = vectorizer.fit_transform(sentences)  # shape: (num_sentences, num_words)


sentence_scores = {}

# Sum TF-IDF weights per sentence (row-wise sum)
for idx, sentence in enumerate(sentences):
    score = tfidf_matrix[idx].sum()
    sentence_scores[sentence] = score

# Generate frequency table and get the most important sentences for the summary
summary = get_summary(sentences, sentence_scores, top_n=2)
print("Özet:\n", summary)







