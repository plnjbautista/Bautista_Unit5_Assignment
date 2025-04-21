import wikipedia
import nltk
import re
import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Download NLTK resources
nltk.download('punkt')

# Topics to fetch
topics = ["Internet of Things", "Artificial Intelligence"]

# Fetch articles from Wikipedia
corpus = [wikipedia.page(topic).content for topic in topics]

# Preprocess: tokenize, clean, lowercase
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = nltk.word_tokenize(text)
    return tokens

# Split article into smaller chunks (paragraph-level)
def split_into_chunks(text, chunk_size=50):
    tokens = preprocess(text)
    return [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size) if len(tokens[i:i + chunk_size]) >= 20]

# Prepare data
tokenized_chunks = []
labels = []

for idx, article in enumerate(corpus):
    chunks = split_into_chunks(article)
    tokenized_chunks.extend(chunks)
    labels.extend([idx] * len(chunks))  # 0 for CS, 1 for AI

# Train Word2Vec model
model = Word2Vec(sentences=tokenized_chunks, vector_size=100, window=5, min_count=1, workers=4)
model.train(tokenized_chunks, total_examples=len(tokenized_chunks), epochs=20)

# Create average word vector for each chunk
def get_doc_vector(tokens, model):
    valid_tokens = [token for token in tokens if token in model.wv]
    if not valid_tokens:
        return np.zeros(model.vector_size)
    return np.mean(model.wv[valid_tokens], axis=0)

X = [get_doc_vector(chunk, model) for chunk in tokenized_chunks]
y = labels

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# Train classifier
clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)

# Debugging: print predictions
print("Actual labels:   ", y_test)
print("Predicted labels:", y_pred)

# Print classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=topics, zero_division=0))
