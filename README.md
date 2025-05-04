# Nlp_ivr_annotator_app.py
A light-weight python driven alternative to Sketch Engine

# NLP IVR Annotator App

A lightweight, Streamlit-based NLP tool for automating the analysis, tagging, concordance search, and semantic clustering of utterances used in Conversational IVR systems (e.g., Kore.ai, Genesys, Dialogflow). Built as a Python-based alternative to Sketch Engine.

## Features

- **POS Tagging & Named Entity Recognition (NER)** using spaCy
- **Lemma Extraction** to normalize training utterances
- **Concordance Search** using NLTK's Text module
- **Semantic Similarity Clustering** via TF-IDF and Cosine Similarity
- **CSV Upload & Export** with annotated data

---

## Setup

### 1. Clone the Repository

bash
git clone https://github.com/YOUR_USERNAME/nlp-ivr-annotator.git
cd nlp-ivr-annotator

### 2. Create and Activate a Virtual Environment

bash
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate

### 3. Install Requirements

bash
pip install -r requirements.txt

# or manualy install dependencies:
pip install streamlit spacy nltk scikit-learn pandas
python -m nltk.downloader punkt stopwords
python -m spacy download en_core_web_sm

## USAGE

### 1. Set up the Streamlit App

bash
streamlit run nlp_ivr_annotator_app.py

### 2. Upload your dataset
# NOTE: The App is expecting a .csv file with at least one column labeled "utterance"

### 3. Explore the Interface
  --
	•	View lemmatized, POS-tagged, and NER-annotated utterances.
	•	Enter a keyword to generate concordance context windows.
	•	Select an utterance index to view top semantic matches via cosine similarity.
	•	Export your annotated CSV.

### License

This project is open-source and provided under the MIT License.

  --

### Maintainer
  --
Matthan Bird — Conversational AI Designer | Data Scientist | Speech UX Engineer
If you find this tool useful, feel free to fork or contribute.

