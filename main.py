from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


# List dokumen yang akan diindeks
documents = [
]

# Buat matriks dokumen-term dengan menggunakan TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# Hitung SVD dari matriks dokumen-term
lsa = TruncatedSVD(n_components=2)
X_lsa = lsa.fit_transform(X)

# Indeks LSI
index = {}
for i, document in enumerate(documents):
    index[document] = X_lsa[i]

# Pencarian
def search(query):
    # Hitung vektor query dengan TfidfVectorizer
    query_vector = vectorizer.transform([query])
    # Hitung latent concept dari query dengan menggunakan SVD
    query_lsa = lsa.transform(query_vector)
    # Inisialisasi skor dan hasil pencarian
    scores = {}
    results = []
    # Bandingkan latent concept dari query dengan latent concept dari setiap dokumen dalam indeks
    for document, lsa_vector in index.items():
        score = cosine_similarity(query_lsa.reshape(1, -1), lsa_vector.reshape(1, -1))[0][0]
        scores[document] = score
        # Tambahkan dokumen ke hasil pencarian jika skornya di atas threshold tertentu
        if score > 0.8:
            results.append(document)
    # Urutkan hasil pencarian berdasarkan skor
    results.sort(key=lambda x: scores[x], reverse=True)
    return results

#sediakan input dan panggil fungsi search
query = input("Masukkan query: ")
results = search(query)
print("Hasil pencarian: ")
for result in results:
    print(result)

#sediakan pilihan untuk mengulang pencarian
while True:
    ulang = input("Apakah anda ingin mencari lagi? (y/n): ")
    if ulang == "y":
        query = input("Masukkan query: ")
        results = search(query)
        print("Hasil pencarian: ")
        for result in results:
            print(result)
    elif ulang == "n":
        break
    else:
        print("Input tidak valid")
