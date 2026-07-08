import requests
import pandas as pd
import time

APP_ID = 4704690
OUTPUT_FILE = "meccha_chameleon_reviews.csv"


def fetch_all_reviews(app_id):
    """Fetch all English reviews from Steam for a given app ID using cursor-based pagination."""
    reviews = []
    cursor = "*"
    total_fetched = 0

    print("Fetching reviews from Steam...")

    while True:
        url = "https://store.steampowered.com/appreviews/"
        params = {
            "json": 1,
            "language": "english",
            "num_per_page": 100,
            "cursor": cursor,
            "filter": "recent",
            "review_type": "all",
            "purchase_type": "all",
        }

        response = requests.get(f"{url}{app_id}", params=params)
        data = response.json()

        # If no more reviews, stop
        batch = data.get("reviews", [])
        if not batch:
            break

        for review in batch:
            reviews.append({
                "review_id": review["recommendationid"],
                "voted_up": review["voted_up"],          # True = positive, False = negative
                "review_text": review["review"],
                "playtime_hours": round(review["author"]["playtime_forever"] / 60, 1),
            })

        total_fetched += len(batch)
        print(f"  Fetched {total_fetched} reviews so far...")

        # Get next cursor for pagination
        cursor = data.get("cursor", "")
        if not cursor:
            break

        # Be polite to Steam API
        time.sleep(0.5)

    return reviews


def main():
    reviews = fetch_all_reviews(APP_ID)

    if not reviews:
        print("No reviews found.")
        return

    df = pd.DataFrame(reviews)

    # Drop reviews with empty text
    df = df[df["review_text"].str.strip() != ""]
    df = df.reset_index(drop=True)

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\nDone! Saved {len(df)} reviews to '{OUTPUT_FILE}'")
    print(df.head(3))


if __name__ == "__main__":
    main()
