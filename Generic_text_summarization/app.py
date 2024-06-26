import streamlit as st
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

# Load SpaCy English model
nlp = spacy.load('en_core_web_sm')
stop_words = list(STOP_WORDS)
punctuation = punctuation + '\n'

# Function to check if input has enough words
def enough_words(text):
    return len(text.split()) >= 50  # Adjust the minimum word count as needed

# Function to summarize text
def text_summarizer(text, percent):
    doc = nlp(text)
    
    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in stop_words and word.text.lower() not in punctuation:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1
    
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word]/max_frequency
    
    # Convert doc.sents to list to calculate length
    sentences = list(doc.sents)
    
    # Adjust select_length based on input length and percentage
    if len(sentences) < 5 and percent > 50:
        # Display a warning if input is too short and percent is high
        st.warning('Input has fewer sentences. Including more information.')
        select_length = len(sentences)  # Include all sentences
    else:
        select_length = int(len(sentences) * (100 - percent) / 100)  # Calculate summary length based on percentage
    
    sentence_scores = {}
    for sent in sentences:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]
    
    summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
    final_summary = ' '.join([word.text for word in summary])
    
    return final_summary

# Streamlit app
def main():
    st.title('Text Summarization App')
    raw_text = st.text_area('Enter your text here:')
    
    # Add a slider for percentage selection
    percent = st.slider('Select percentage of text to summarize', 10, 100, 50, 1)
    
    if st.button('Summarize'):
        if raw_text:
            if enough_words(raw_text):
                summary = text_summarizer(raw_text, percent)
                st.subheader('Summary:')
                # Display summarized text in a container with paragraph format
                with st.container():
                    st.markdown(f'{summary}', unsafe_allow_html=True)
            else:
                st.warning('Input has fewer words. Please provide more text for summarization.')

if __name__ == '__main__':
    main()
