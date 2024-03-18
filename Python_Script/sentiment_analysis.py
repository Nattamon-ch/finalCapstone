# Import necessary libraries
import spacy  # For NLP tasks
from spacy.lang.en.stop_words import STOP_WORDS  # To access English stop words
import pandas as pd  # For data manipulation
from textblob import TextBlob  # For sentiment analysis

# Load spaCy's English-language models
nlp = spacy.load('en_core_web_sm')  # Small model for preprocessing
nlp2 = spacy.load('en_core_web_md')  # Medium model for more accurate similarity calculations

# Load data from a CSV file
amazon_review_df = pd.read_csv("/Users/preem.ds/Downloads/14-003-1 Capstone Project - NLP Applications/amazon_product_reviews.csv")

# Remove rows with missing review text to ensure data quality
clean_data = amazon_review_df.dropna(subset=['reviews.text'])

# Define a function to preprocess text data
def preprocess_text(text):
    doc = nlp(text)  # Process text to create a spaCy Doc object
    # Generate a list of tokens, converted to lowercase and stripped of whitespace, excluding stop words
    tokens = [token.text.lower().strip() for token in doc if not token.is_stop]
    return ' '.join(tokens)  # Rejoin tokens into a single string and return

# Apply preprocessing to the reviews
clean_data['reviews.text'] = clean_data['reviews.text'].apply(preprocess_text)

# Define a function to calculate the polarity of text using TextBlob
def get_polarity(text):
    return TextBlob(text).sentiment.polarity

# Define a function to determine the sentiment based on polarity
def get_sentiment(polarity):
    if polarity > 0:
        return 'Positive'
    elif polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

# Add columns for polarity and sentiment to the DataFrame
clean_data['Polarity'] = clean_data['reviews.text'].apply(get_polarity)
clean_data['Sentiment'] = clean_data['Polarity'].apply(get_sentiment)

# Save the processed and analysed data to a new CSV file
clean_data.to_csv("/Users/preem.ds/Downloads/14-003-1 Capstone Project - NLP Applications/analyzed_amazon_product_reviews.csv", index=False)

# Define a function for sentiment analysis that returns the sentiment of a review
def sentiment_analysis(review):
    polarity = get_polarity(review)  # Calculate polarity
    sentiment = get_sentiment(polarity)  # Determine sentiment from polarity
    return sentiment

# Initialise variables to track the most and least similar review pairs within the first 20 reviews
most_similar = {'score': 0, 'pair': (None, None)}
least_similar = {'score': 1, 'pair': (None, None)}

# Compare similarity between review pairs
for i in range(min(25, len(clean_data))):
    for j in range(i + 1, min(25, len(clean_data))):
        doc1 = nlp2(clean_data['reviews.text'].iloc[i])  # Process ith review
        doc2 = nlp2(clean_data['reviews.text'].iloc[j])  # Process jth review
        score = doc1.similarity(doc2)  # Calculate similarity score
        # Update most similar pair if current score is higher
        if score > most_similar['score']:
            most_similar['score'] = score
            most_similar['pair'] = (i, j)
        # Update least similar pair if current score is lower
        if score < least_similar['score']:
            least_similar['score'] = score
            least_similar['pair'] = (i, j)

# Print the indices and scores of the most and least similar review pairs
print(f"Most similar reviews are at index {most_similar['pair']} with a score of {most_similar['score']}")
print(f"Least similar reviews are at index {least_similar['pair']} with a score of {least_similar['score']}")

# Retrieve and print the most and least similar review pairs for analysis
most_SR_pair = clean_data.iloc[list(most_similar['pair'])]['reviews.text']
least_SR_pair = clean_data.iloc[list(least_similar['pair'])]['reviews.text']

# Print sentiment analysis for the most and least similar review pairs
for review in most_SR_pair:
    print(f"Review: {review}")
    print(f"Sentiment: {sentiment_analysis(review)}")
for review in least_SR_pair:
    print(f"Review: {review}")
    print(f"Sentiment: {sentiment_analysis(review)}")
