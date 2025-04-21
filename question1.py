'''
    1.(20 points) Using Wikipedia as the corpus, obtain 5 different topics that 
    will serve as your documents, and create a term-document matrix. You can use 
    the shared code on GitHub as a reference.
    '''

import wikipedia
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# 5 topics
topics = [
    "Computer Science",
    "Artificial Intelligence",
    "Natural Language Processing",
    "Apple",
    "Internet of Things"
]

# Fetch Wikipedia Content
documents = []
for topic in topics:
    try:
        content = wikipedia.page(topic).content
        documents.append(content)
    except Exception as e:
        print(f"Error fetching {topic}: {e}")
        documents.append("")

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) # Remove punctuation
    return text

documents_clean = [preprocess(doc) for doc in documents]

# a. Term-Document Matrix using Raw Frequency
count_vectorizer = CountVectorizer(stop_words='english')
X_counts = count_vectorizer.fit_transform(documents_clean)
df_counts = pd.DataFrame(X_counts.toarray(), columns=count_vectorizer.get_feature_names_out(), index=topics)
print("Raw Frequency Matrix: ")
print(df_counts)

# Save Raw Frequency Matrix to CSV
df_counts.to_csv("term_document_counts.csv")


# b. Term-Document Matrix using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = tfidf_vectorizer.fit_transform(documents_clean)
df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out(), index=topics)
print("\nTF-IDF Matrix: ")
print(df_tfidf)

# Save TF-IDF Matrix to CSV
df_tfidf.to_csv("term_document_tfidf.csv")