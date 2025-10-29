import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def load_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]

def load_stopwords(stopword_path):
    with open(stopword_path, 'r', encoding='utf-8') as f:
        return set([line.strip().lower() for line in f if line.strip()])


def clean_sentence(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s]', '', sentence)  # belgilarni olib tashlash
    return sentence

def remove_stopwords(sentence, stopwords):
    words = sentence.split()
    return ' '.join([word for word in words if word not in stopwords])


def cluster_sentences(sentences, stopwords, k):
    cleaned = [remove_stopwords(clean_sentence(sent), stopwords) for sent in sentences]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(cleaned)
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X)
    clustered = [[] for _ in range(k)]
    for i, label in enumerate(labels):
        clustered[label].append(sentences[i])  # asl (tozalamasdan) gapni yozamiz
    return clustered


def save_result(clustered, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, group in enumerate(clustered):
            f.write(f"=== KLASTER {i+1} ===\n")
            for sentence in group:
                f.write(sentence + "\n")
            f.write("\n")

if __name__ == "__main__":
    gap_fayl = "gaplar.txt"  # Matnli gaplar fayli
    stopwords_fayl = "stop_words_uz.txt"  # Stopword fayli
    natija_fayl = "klasterlangan_natija.txt"

    # Klaster sonini kiritish
    k = int(input("Nechta klasterga ajratilsin? "))

    # Fayllarni yuklab, klasterlashni boshlaymiz
    if os.path.exists(gap_fayl) and os.path.exists(stopwords_fayl):
        sentences = load_sentences(gap_fayl)
        stopwords = load_stopwords(stopwords_fayl)
        clustered = cluster_sentences(sentences, stopwords, k)
        save_result(clustered, natija_fayl)
        print(f"✅ {k} ta klasterga ajratildi. Natija '{natija_fayl}' faylida saqlandi.")
    else:
        print(" Fayllar topilmadi.")
