import pandas as pd
import re
from textblob import TextBlob
from nltk.corpus import stopwords
import nltk

# Download stopwords if not already downloaded
nltk.download("stopwords", quiet=True)

INPUT_FILE = "meccha_chameleon_reviews.csv"
OUTPUT_FILE = "meccha_chameleon_sentiment.csv"

STOP_WORDS = set(stopwords.words("english"))


def clean_text(text):
    """Clean review text: lowercase, remove punctuation, filter stop words."""
    # Lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r"[^a-z\s]", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Filter out stop words
    words = text.split()
    words = [word for word in words if word not in STOP_WORDS]

    return " ".join(words)


def get_sentiment_label(polarity):
    """Convert TextBlob polarity score to a sentiment label."""
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"


def analyze_sentiment(text):
    """Return polarity score for a given text using TextBlob."""
    return TextBlob(text).sentiment.polarity


def main():
    # Load the dataset
    print("Loading reviews...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Total reviews loaded: {len(df)}")

    # Clean the review text
    print("\nCleaning text...")
    df["cleaned_review"] = df["review_text"].fillna("").astype(str).apply(clean_text)

    # Drop rows where cleaned text is empty
    df = df[df["cleaned_review"].str.strip() != ""]
    df = df.reset_index(drop=True)
    print(f"Reviews after cleaning: {len(df)}")

    # Calculate sentiment polarity score for each review
    print("\nAnalyzing sentiment (this may take a moment)...")
    df["polarity_score"] = df["cleaned_review"].apply(analyze_sentiment)

    # Label each review as Positive, Negative, or Neutral
    df["sentiment"] = df["polarity_score"].apply(get_sentiment_label)

    # Save results to a new CSV
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\nResults saved to '{OUTPUT_FILE}'")

    # Print final statistics
    print("\n--- Sentiment Analysis Results ---")
    counts = df["sentiment"].value_counts()
    total = len(df)

    for label, count in counts.items():
        percentage = (count / total) * 100
        print(f"  {label}: {count} reviews ({percentage:.1f}%)")

    print(f"\nAverage polarity score: {df['polarity_score'].mean():.3f}")
    print(f"  (-1.0 = very negative, 0.0 = neutral, +1.0 = very positive)")


if __name__ == "__main__":
    main()