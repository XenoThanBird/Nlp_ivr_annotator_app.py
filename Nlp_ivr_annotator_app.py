import streamlit as st
import pandas as pd
import spacy
import nltk
from nltk.text import Text
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup
nltk.download('punkt')
nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

# Concordance
def generate_concordance(text, keyword, window=5):
    tokens = nltk.word_tokenize(text)
    concord_text = Text(tokens)
    return concord_text.concordance_list(keyword, width=2 * window + 1, lines=5)

# Tagging
def tag_and_label(text):
    doc = nlp(text)
    return {
        "lemmas": [token.lemma_ for token in doc if not token.is_punct],
        "pos_tags": [(token.text, token.pos_) for token in doc],
        "entities": [(ent.text, ent.label_) for ent in doc.ents]
    }

# Cosine Similarity
def compute_similarity(text_series):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(text_series)
    return cosine_similarity(tfidf_matrix)

# Get Top Similar Utterances
def get_top_similar(index, matrix, texts, top_n=3):
    sim_scores = list(enumerate(matrix[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    return [(texts[i], round(score, 2)) for i, score in sim_scores[1:top_n+1]]

# App UI
st.title("Conversational IVR NLP Annotator")
st.markdown("This tool analyzes utterances for tagging, concordance, and similarity clustering.")

uploaded_file = st.file_uploader("Upload a CSV with an `utterance` column", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'utterance' not in df.columns:
        st.error("CSV must contain a column named 'utterance'")
    else:
        st.success("Data loaded successfully!")

        # Tag and label
        st.subheader("POS & Entity Tagging")
        df['annotations'] = df['utterance'].apply(tag_and_label)
        st.dataframe(df[['utterance', 'annotations']])

        # Concordance
        st.subheader("Concordance Search")
        keyword = st.text_input("Enter keyword for concordance")
        if keyword:
            df['concordance'] = df['utterance'].apply(lambda x: generate_concordance(x, keyword) if keyword in x else [])
            st.write(df[['utterance', 'concordance']])

        # Similarity Matrix
        st.subheader("Semantic Similarity")
        similarity_matrix = compute_similarity(df['utterance'])
        sim_index = st.number_input("Choose an utterance index to find top matches", min_value=0, max_value=len(df)-1, step=1)
        if st.button("Show Top Similar Utterances"):
            top_matches = get_top_similar(sim_index, similarity_matrix, df['utterance'])
            st.write(f"Top matches for: *{df['utterance'][sim_index]}*")
            for match, score in top_matches:
                st.write(f"- {match} ({score})")

        # Download
        st.download_button(
            "Download Annotated CSV",
            df.to_csv(index=False).encode(),
            file_name="annotated_utterances.csv",
            mime="text/csv"
        )
