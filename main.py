from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

#route untuk menampilkan halaman utama dalam folder template dengan nama index.html
@app.route('/')
def home():
    return render_template('index.html')



# List dokumen yang akan diindeks
documents = [
    "aubrey adalah orang yang cantik",
    "wildan adalah orang yang gateng"
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
    # Jika result kosong, tambahkan dokumen dengan skor diatas 0.3
    if not results:
        for document, score in scores.items():
            if score > 0.3:
                results.append(document)
    #jika masih kosong, Tambahkan Result not found
    if not results:
        results.append("Result not found")
    # Kembalikan hasil pencarian
    return results

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        query = request.form['query']
        results = search(query)
        return render_template('index.html', results=results)

if __name__ == "__main__":
    app.run(debug=True)