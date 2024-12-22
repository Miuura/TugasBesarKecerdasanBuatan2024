import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv
from datetime import datetime
import joblib
import re
import emoji
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Cleanning Data
def load_stopwords():
    stopwords_path = r'c:\Users\Valent\AppData\Roaming\nltk_data\corpora\stopwords\english'
    with open(stopwords_path, 'r') as file:
        stopwords_list = file.read().splitlines()
    return set(stopwords_list)

vectorizer = joblib.load('tfidf_vectorizer.pkl')

slang_dict = {
    "u": "you",
    "omg": "oh my god",
    "bff": "best friend forever",
    "gonna": "going to",
    "wanna": "want to",
    "idk": "i don't know",
    "brb": "be right back",
}

stop_words = load_stopwords()

lemmatizer = WordNetLemmatizer()
def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]

def normalize_text(text):
  text = text.lower()
  text = emoji.demojize(text)
  text = re.sub(r'@[A-Za-z0-9_]+', '', text)
  text = re.sub(r'#\w+', '', text)
  text = re.sub(r'rt[\s]+', '', text)
  text = re.sub(r'https?://\S+', '', text)
  text = re.sub(r'[^A-Za-z0-9\' ]', '', text)
  text = re.sub(r'\s+', ' ', text).strip()

  # Normalisasi slang
  words = text.split()
  normalized_words = [slang_dict[word] if word in slang_dict else word for word in words]

  return ' '.join(normalized_words)

def sentiment_analysis(text):
  text = normalize_text(text)
  text = word_tokenize(text)
  text = lemmatize_tokens(text)
  filtered_tokens = [word for word in text if word.lower() not in stop_words]
  text = ' '.join(filtered_tokens)
  clean_text = vectorizer.transform([text])
  return clean_text

load_dotenv()
twitter_auth_token = os.getenv("TWITTER_AUTH_TOKEN")

dt = joblib.load("DecisionTree_Model.joblib")
nb = joblib.load("MultinomialNB_Model.joblib")

# Title and description
st.title("Sentiment Analysis for Suicidal Detection 🪽 ")
# st.write("Analyze tweets for potential suicidal tendencies. This tool crawls data based on your keywords and performs sentiment analysis.")
st.markdown("""
### :books: Tugas Besar Artificial Intelligence
- **Valentino Hartanto** - 1301223020
- **Gede Bagus Krishnanditya Merta** - 1301223088 
- **Muhammad Azmi** - 1301223282

### :page_facing_up: Deskripsi
Aplikasi ini dirancang untuk menganalisis tweet guna mendeteksi potensi kecenderungan bunuh diri. Dengan memanfaatkan crawling data, aplikasi ini memungkinkan pengguna untuk mencari tweet berdasarkan keywords tertentu dalam rentang waktu yang ditentukan. Data yang diperoleh kemudian diproses melalui sentiment analysis untuk mengidentifikasi pola yang mengindikasikan risiko bunuh diri.
""")

st.sidebar.header("Crawl Parameters 🔍 ")
st.sidebar.markdown("💡 Designed to help you analyze tweets effectively!")

# User input
keyword = st.sidebar.text_input("🔑 Keyword for data crawling: ", "suicide")
since_date = st.sidebar.date_input("Since (Start Date):")
until_date = st.sidebar.date_input("Until (End Date):")
num_results = st.sidebar.slider("Number of results to crawl:", 10, 500, 100)


# Button to start crawling and analysis
if st.sidebar.button("Start Crawling and Analysis 🚀"):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f'crawled_data_{timestamp}.csv'

    search_keyword = f'{keyword} since:{since_date} until:{until_date} lang:en'

    # tweet harvesting command
    crawling_message = st.empty()
    crawling_message.write("⏳ Crawling data, please wait...")
    os.system(f'npx -y tweet-harvest@2.6.1 -o "{filename}" -s "{search_keyword}" --tab "LATEST" -l {num_results} --token {twitter_auth_token}')

    # Read the CSV file into a pandas DataFrame
    try:
        crawled_data = pd.read_csv("./tweets-data/"+filename, delimiter=",")
        crawling_message.success("✅ Data successfully crawled!")
        st.write("### 📁 Crawled Data:")
        selected_columns = ["created_at", "favorite_count", "full_text", "retweet_count", "reply_count", "username", "tweet_url"]
        crawled_data = crawled_data[selected_columns]
        st.dataframe(crawled_data)

        # st.write("🎯 Performing sentiment analysis...")
        crawled_data["clean_full_text"] = crawled_data["full_text"].apply(sentiment_analysis)
        crawled_data["Sentiment"] = crawled_data["clean_full_text"].apply(nb.predict)

        # Sentiment Analysis Summary
        # sentiment_series = pd.Series(crawled_data["Sentiment"])
        # sentiment_counts = sentiment_series.value_counts()
        # potential_percent = (sentiment_counts.get("[Potential Suicide post ]", 0) / len(crawled_data)) * 100
        
        # not_potential_percent = (sentiment_counts.get("[Not Suicide post]", 0) / len(crawled_data)) * 100

        value_counts = crawled_data['Sentiment'].value_counts(normalize=True) * 100
        categories = value_counts.index
        percentages = value_counts.values

        st.write("### 📊 Sentiment Analysis Results:")
        st.write(f"Potential Suicidal Posts: {percentages[1]:.2f}%")
        st.write(f"Not Potential Suicidal Posts: {percentages[0]:.2f}%")

        kategori = ["Not Potential Suicidal Posts", "Potential Suicidal Posts"]
        persentase = [percentages[0], percentages[1]]

        # Bar plot
        fig, ax = plt.subplots(figsize=(8, 6))  # Hanya satu plot
        sns.barplot(x=kategori, y=persentase, palette="pastel", ax=ax)
        ax.set_ylabel("Percentage (%)", fontsize=12)
        ax.set_xlabel("Suicide Categories", fontsize=12)
        ax.set_title("Distribution of Suicide Categories", fontsize=14, weight='bold')
        ax.set_ylim(0, 100)
        ax.tick_params(axis='x', labelsize=10)

        st.pyplot(fig)

        st.write("### 📁 Predict Data")
        selected_columns = ["created_at", "full_text", "username", "Sentiment"]
        crawled_data = crawled_data[selected_columns]
        st.dataframe(crawled_data)


    except FileNotFoundError:
        st.error("❌ Failed to crawl data. Please check your inputs and try again.")

st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Built with ❤️ by Kelompok 8**  
    """
)