'''
    3.(10 points) Using cosine similarity, compare two documents and find out which of the documents is most similar.
    For this one, I am not sure if I understand the question correctly so I tried 2 approaches: 
    Comparing specific documents and making list of pairings from the 5 topics from number 1 and comparing them.
    '''
import wikipedia
import re
import math
from itertools import combinations
from collections import defaultdict

# Preprocess the text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Create document vector from term frequencies
def create_document_vector(doc, vocab):
    term_freq = defaultdict(int)
    for term in doc.split():
        if term in vocab:
            term_freq[term] += 1
    return term_freq

# Cosine similarity function
def cosine_similarity(vec1, vec2, vocab):
    dot_product = sum(vec1[term] * vec2[term] for term in vocab)
    vec1_len = math.sqrt(sum(vec1[term]**2 for term in vocab))
    vec2_len = math.sqrt(sum(vec2[term]**2 for term in vocab))
    if vec1_len == 0 or vec2_len == 0:
        return 0.0
    return dot_product / (vec1_len * vec2_len)

# Compare two specific documents
def compare_specific(doc1_name, doc2_name):
    try:
        doc1_text = preprocess_text(wikipedia.page(doc1_name).content)
        doc2_text = preprocess_text(wikipedia.page(doc2_name).content)
    except Exception as e:
        print(f"Error fetching one of the pages: {e}")
        return

    vocab = set(doc1_text.split()).union(set(doc2_text.split()))
    vec1 = create_document_vector(doc1_text, vocab)
    vec2 = create_document_vector(doc2_text, vocab)
    sim = cosine_similarity(vec1, vec2, vocab)
    print(f"Cosine Similarity between '{doc1_name}' and '{doc2_name}': {sim:.4f}")

# Compare all pairs from a list and show the most similar pair
def compare_all_pairs(topics):
    documents = {}
    for topic in topics:
        try:
            content = wikipedia.page(topic).content
            documents[topic] = preprocess_text(content)
        except Exception as e:
            print(f"Error fetching '{topic}': {e}")

    max_similarity = 0.0
    most_similar_pair = ("", "")

    print("\nCosine Similarity Between Topic Pairs:\n")
    for topic1, topic2 in combinations(documents.keys(), 2):
        doc1_text = documents[topic1]
        doc2_text = documents[topic2]
        vocab = set(doc1_text.split()).union(set(doc2_text.split()))
        vec1 = create_document_vector(doc1_text, vocab)
        vec2 = create_document_vector(doc2_text, vocab)
        sim = cosine_similarity(vec1, vec2, vocab)
        print(f"{topic1} ↔ {topic2}: {sim:.4f}")

        if sim > max_similarity:
            max_similarity = sim
            most_similar_pair = (topic1, topic2)

    print("\nMost Similar Pair:")
    print(f"{most_similar_pair[0]} ↔ {most_similar_pair[1]} with similarity {max_similarity:.4f}")

# Run the comparisons
print("Cosine Similarity Between Specific Documents:")
compare_specific("Artificial Intelligence", "Internet of Things")

topics = [
    "Computer Science",
    "Artificial Intelligence",
    "Natural Language Processing",
    "Apple",
    "Internet of Things"
]

print("\nComparing All Topic Pairs:")
compare_all_pairs(topics)
