import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle
from flask import Flask, request, render_template

app = Flask(__name__, template_folder='templates')

# Load the pre-trained model and TF-IDF vectorizer
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Load and preprocess data
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
df.columns = ["label", "message"]

# Preprocessing function
def preprocess_content(text):
    stemmer = PorterStemmer()
    nopunc = ''.join([char for char in text if char not in string.punctuation])
    words = word_tokenize(nopunc.lower())
    nostop = [stemmer.stem(word) for word in words if word not in stopwords.words('english') and word.isalpha()]
    return ' '.join(nostop)

# Apply preprocessing
df['cleaned_text'] = df['message'].apply(preprocess_content)

# Vectorize data
X = tfidf.transform(df['cleaned_text'])
y = df['label']

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['text']
    cleaned_input = preprocess_content(input_text)
    X_new = tfidf.transform([cleaned_input])
    prediction = rf_model.predict(X_new)[0]
    return render_template('index.html', prediction=prediction, input_text=input_text)

if __name__ == '__main__':
    app.run(debug=True)
