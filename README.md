# Meccha Chameleon — Steam Review Sentiment Analysis

An NLP pipeline that fetches all English Steam reviews for [Meccha Chameleon](https://store.steampowered.com/app/4704690/MECCHA_CHAMELEON/) and runs sentiment analysis on them — so I didn't have to read 31,665 reviews myself.

## What it does

1. Fetches all English reviews from the Steam API
2. Cleans the text (lowercase, punctuation removal, stop word filtering)
3. Calculates a sentiment polarity score for each review using TextBlob
4. Labels each review as Positive, Negative, or Neutral
5. Generates a dashboard visualizing the results

## Results

| Sentiment | Count | Percentage |
|-----------|-------|------------|
| Positive  | 14,387 | 46.8% |
| Neutral   | 12,629 | 41.1% |
| Negative  | 3,727  | 12.1% |

Average polarity score: **0.126** (slightly positive)

## Files

- `fetch_reviews.py` — fetches reviews from the Steam API and saves to CSV
- `sentiment_analysis.py` — cleans text and runs sentiment analysis
- `visualize_sentiment.py` — generates the results dashboard

## Setup

```bash
pip install requests pandas textblob nltk matplotlib
```

```bash
python fetch_reviews.py
python sentiment_analysis.py
python visualize_sentiment.py
```

## Built with

- [Steam Web API](https://store.steampowered.com/appreviews/)
- [TextBlob](https://textblob.readthedocs.io/)
- [NLTK](https://www.nltk.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)

---

*i2i Academy — Natural Language Processing assignment*
