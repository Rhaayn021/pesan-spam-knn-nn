import streamlit as st
import pandas as pd
import pickle
import re
from nltk.corpus import stopwords
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib.pyplot as plt

# Membuat instance stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Load key_norm.csv
key_norm = pd.read_csv('key_norm.csv')

nltk.download('stopwords')

#FITUR STATISTIK
# Fungsi preprocessing
def casefolding(text):
    text = text.lower()
    text = re.sub(r'http[s]?:?://\S+|www\.\S+(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
    text = re.sub(r'[-]?[0-9]+', ' ', text)
    text = re.sub(r'[+]?', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text

def text_normalize(text):
    text = ' '.join([key_norm[key_norm['singkat'] == word]['hasil'].values[0]
                     if (key_norm['singkat'] == word).any()
                     else word for word in text.split()])
    text = str.lower(text)
    return text

def remove_stop_word(text):
    stopwords_indo = stopwords.words('indonesian') + ['tsel', 'gb', 'rb', 'btw']
    clean_words = []
    text = text.split()
    for word in text:
        if word not in stopwords_indo:
            clean_words.append(word)
    return " ".join(clean_words)

def stemming(text):
    text = stemmer.stem(text)
    return text

def preprocess_text(text):
    text = casefolding(text)
    text = text_normalize(text)
    text = remove_stop_word(text)
    text = stemming(text)
    return text

#FITUR LINGUISTIK
# Fungsi untuk menghitung panjang kalimat
def sentence_length(text):
    # Memisahkan teks menjadi kata-kata berdasarkan spasi
    words = text.split()
    # Menghitung dan mengembalikan jumlah kata
    return len(words)

# Fungsi untuk menghitung jumlah huruf besar di awal kata
def count_capitalized_words(text):
    return sum(1 for word in re.findall(r'\b[A-Z][a-z]*\b', text))

# Fungsi untuk menghitung jumlah kata cue-phrase promo dan penipuan
def load_cue_phrases(filename):
    return pd.read_csv(filename)['teks'].tolist()

cue_phrases_promo = load_cue_phrases('cue_phrase_promo.csv')
cue_phrases_fraud = load_cue_phrases('cue_phrase_penipuan.csv')

def count_cue_phrases(text, cue_phrases):
    text = str.lower(text)
    return sum(1 for word in text.split() if word in cue_phrases)


with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

st.title('Klasifikasi SMS Spam')

models = ['KNN Statistik', 'NN Statistik', 'KNN Linguistik', 'NN Linguistik']

selected_model = st.radio('Select a model', models)

user_input = st.text_area("Masukkan teks di sini")

# HTML/CSS styling untuk membuat box dengan background merah
error_box = """
<div style="background-color:#841D0F;padding:10px;border-radius:5px;margin:10px 0">
    <p style="color:white;margin:0">Spam Penipuan</p>
</div>
"""

def convert_numbers_to_letters(text):
    # Kamus konversi angka ke huruf
    conversion_dict = {
        '0': 'o',
        '1': 'i',
        '3': 'e',
        '4': 'a',
        '5': 's',
        '6': 'g',
        '7': 't',
        '8': 'b',
        '9': 'p'
    }
    
    # Menghapus informasi sensitif seperti nomor rekening
    text = re.sub(r'\b\d{10,}\b', '', text)

    # Mengganti angka dengan huruf sesuai kamus konversi
    for num, letter in conversion_dict.items():
        text = text.replace(num, letter)
    text = text.lower()
    return text

user_input = convert_numbers_to_letters(user_input)

st.write("Inputan diubah apabila menerima pesan angka :",user_input)

if selected_model == "KNN Linguistik" :
    # Memuat model dan TfidfVectorizer
    with open('model_knn_linguistik.pkl', 'rb') as model_file:
        model_knn_linguistik = pickle.load(model_file)

    # Kalkulasi fitur linguistik
    text_features = {
        'sentence_length': sentence_length(user_input),
        'capitalized_words': count_capitalized_words(user_input),
        'cue_phrases_promo': count_cue_phrases(user_input, cue_phrases_promo),
        'cue_phrases_fraud': count_cue_phrases(user_input, cue_phrases_fraud)
    }
    # Siapkan fitur untuk prediksi dalam format array 2D
    features_linguistic = [list(text_features.values())]

    prediksi = model_knn_linguistik.predict(features_linguistic)

    if not user_input :
        detection = "Inputan tidak boleh kosong"
        st.error(detection)
    else:
        # Output prediksi
        kelas = ["Normal", "Spam Penipuan", "Spam Promo"]
        detection = kelas[prediksi[0]]
        if prediksi[0]==1 :
            # Menggunakan HTML di dalam st.markdown dan mengizinkan HTML
            st.markdown(error_box, unsafe_allow_html=True)
        else :
            st.success(f"{detection}")

        st.subheader('KNN Linguistik Classification Report')
        with open('classification_report_model/report_knn_linguistik.txt', 'r') as file:
            knn_report = file.read()
        st.text_area("KNN Report", knn_report, height=200)

        probabilitas = model_knn_linguistik.predict_proba(features_linguistic)
        kelas = ["Normal", "Spam Penipuan", "Spam Promo"]
        probabilitas_bulat = [round(p, 2) for p in probabilitas[0]]
        st.write("Probabilitas: ", dict(zip(kelas, probabilitas_bulat)))

elif selected_model == "NN Linguistik" :
    # Memuat model dan TfidfVectorizer
    with open('model_nn_linguistik.pkl', 'rb') as model_file:
        model_nn_linguistik = pickle.load(model_file)

    text_baru_linguistik = {}
    text_baru_linguistik['sentence_length'] = sentence_length(user_input)
    text_baru_linguistik['capitalized_words'] = count_capitalized_words(user_input)
    text_baru_linguistik['cue_phrases_promo'] = count_cue_phrases(user_input, cue_phrases_promo)
    text_baru_linguistik['cue_phrases_fraud'] = count_cue_phrases(user_input, cue_phrases_fraud)

    # Siapkan fitur untuk prediksi dalam format array 2D
    features_linguistic = [list(text_baru_linguistik.values())]

    prediksi = model_nn_linguistik.predict(features_linguistic)

    if not user_input :
        detection = "Inputan tidak boleh kosong"
        st.error(detection)
    else:
        # Output prediksi
        kelas = ["Normal", "Spam Penipuan", "Spam Promo"]
        detection = kelas[prediksi[0]]
        if prediksi[0]==1 :
            st.markdown(error_box, unsafe_allow_html=True)
        else :
            st.success(f"{detection}")
        
        st.subheader('NN Linguistik Classification Report')
        with open('classification_report_model/report_nn_linguistik.txt', 'r') as file:
            knn_report = file.read()
        st.text_area("KNN Report", knn_report, height=200)
        
        probabilitas = model_nn_linguistik.predict_proba(features_linguistic)
        kelas = ["Normal", "Spam Penipuan", "Spam Promo"]
        probabilitas_bulat = [round(p, 2) for p in probabilitas[0]]
        st.write("Probabilitas: ", dict(zip(kelas, probabilitas_bulat)))

elif selected_model == "KNN Statistik" :
    # Memuat model dan TfidfVectorizer
    with open('model_knn_statistik.pkl', 'rb') as model_file:
        model_knn_statistik = pickle.load(model_file)

    # Preprocess input pengguna
    preprocessed_text = preprocess_text(user_input)
    # Vectorize teks
    vectorized_text = tfidf_vectorizer.transform([preprocessed_text])
    # Melakukan prediksi
    prediksi = model_knn_statistik.predict(vectorized_text)

    if not user_input :
        detection = "Inputan tidak boleh kosong"
        st.error(detection)
    else:
        # Output prediksi
        kelas = ["Normal", "Spam Penipuan", "Spam Promo"]
        detection = kelas[prediksi[0]]
        if prediksi[0]==1 :
            st.markdown(error_box, unsafe_allow_html=True)
        else :
            st.success(f"{detection}")

        st.subheader('KNN Statistik Classification Report')
        with open('classification_report_model/report_knn_statistik.txt', 'r') as file:
            knn_report = file.read()
        st.text_area("KNN Report", knn_report, height=200)
        
        probabilitas = model_knn_statistik.predict_proba(vectorized_text)
        kelas = ["Normal", "Spam Penipuan", "Spam Promo"]
        probabilitas_bulat = [round(p, 2) for p in probabilitas[0]]
        st.write("Probabilitas: ", dict(zip(kelas, probabilitas_bulat)))

elif selected_model == "NN Statistik" :
    # Memuat model dan TfidfVectorizer
    with open('model_nn_statistik.pkl', 'rb') as model_file:
        model_nn_statistik = pickle.load(model_file)


    # Preprocess input pengguna
    preprocessed_text = preprocess_text(user_input)
    # Vectorize teks
    vectorized_text = tfidf_vectorizer.transform([preprocessed_text])
    # Melakukan prediksi
    prediksi = model_nn_statistik.predict(vectorized_text)

    if not user_input :
        detection = "Inputan tidak boleh kosong"
        st.error(detection)
    else:
        # Output prediksi
        kelas = ["Normal", "Spam Penipuan", "Spam Promo"]
        detection = kelas[prediksi[0]]
        if prediksi[0]==1 :
            st.markdown(error_box, unsafe_allow_html=True)
        else :
            st.success(f"{detection}")

        st.subheader('Neural Network Statistik Classification Report')
        with open('classification_report_model/report_nn_statistik.txt', 'r') as file:
            nn_report = file.read()
        st.text_area("NN Report", nn_report, height=200)

        probabilitas = model_nn_statistik.predict_proba(vectorized_text)
        kelas = ["Normal", "Spam Penipuan", "Spam Promo"]
        probabilitas_bulat = [round(p, 2) for p in probabilitas[0]]
        st.write("Probabilitas: ", dict(zip(kelas, probabilitas_bulat)))
        
        # Menampilkan grafik
        fig, ax = plt.subplots()
        ax.bar(kelas, probabilitas[0], color=['#40E0D0', '#800000', '#40E0D0'])
        plt.xlabel('Kategori')
        plt.ylabel('Probabilitas')
        plt.title('Probabilitas Prediksi Klasifikasi')
        st.pyplot(fig)
