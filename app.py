from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load and preprocess the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('Covid-19_Tweets.csv')
    return data

# Step 2: Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to calculate sentiment scores
def analyze_sentiment(tweet):
    score = analyzer.polarity_scores(tweet)
    return score['compound'], score['pos'], score['neg'], score['neu']

# Step 3: Prepare the app
st.title("COVID-19 Tweets Sentiment Analysis")

data = load_data()
if st.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.write(data)

# Analyze sentiment for each tweet
data['compound'], data['positive'], data['negative'], data['neutral'] = zip(*data['Tweets'].apply(analyze_sentiment))

# Display sentiment results including polarity and subjectivity scores
st.subheader("Sentiment Analysis Results")
st.write(data[['Tweets', 'compound', 'positive', 'negative', 'neutral']])

# Predict sentiment category based on compound score
def get_sentiment_label(compound):
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

data['Sentiment'] = data['compound'].apply(get_sentiment_label)

# Show sentiment distribution
st.subheader("Sentiment Distribution")
sentiment_counts = data['Sentiment'].value_counts()
st.bar_chart(sentiment_counts)

# Step 4: Exploratory Data Analysis (EDA)
st.subheader("Exploratory Data Analysis (EDA)")

# Distribution of emotions in the dataset
emotion_counts = data['Emotions'].value_counts()
st.bar_chart(emotion_counts)

# Visualize the distribution of sentiments by emotion type
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Emotions', hue='Sentiment', palette='Set2')
plt.title('Distribution of Sentiments by Emotion Type')
plt.xticks(rotation=45)
plt.xlabel('Emotions')
plt.ylabel('Count')
plt.legend(title='Sentiment')
st.pyplot(plt)

# Additional analysis: Correlation heatmap (if applicable)
if 'Emotions' in data.columns and len(data) > 0:
    emotion_dummies = pd.get_dummies(data['Emotions'])
    correlation_matrix = emotion_dummies.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Heatmap of Emotions')
    st.pyplot(plt)

# Show summary statistics of the dataset
st.subheader("Summary Statistics")
st.write(data.describe())

# Step 5: Model Selection and Input Text for Prediction
st.subheader("Predict Sentiment for Your Text")

# Slider for model selection (for now we only have one model - VADER)
model_choice = st.selectbox("Select a Model", ["VADER"])

if model_choice == "VADER":
    user_input = st.text_area("Enter your text here:")
    
    if st.button("Predict Sentiment"):
        if user_input:
            compound_score, pos_score, neg_score, neu_score = analyze_sentiment(user_input)
            sentiment_label = get_sentiment_label(compound_score)
            
            # Display results to user
            st.write(f"**Compound Score:** {compound_score:.4f}")
            st.write(f"**Positive Score:** {pos_score:.4f}")
            st.write(f"**Negative Score:** {neg_score:.4f}")
            st.write(f"**Neutral Score:** {neu_score:.4f}")
            st.write(f"**Predicted Sentiment:** {sentiment_label}")
        else:
            st.warning("Please enter some text to analyze.")

