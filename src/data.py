"""Dataset helpers. Both notebooks call make_split() so the classical and
deep-learning models are evaluated on exactly the same test tweets."""
import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = ["negative", "neutral", "positive"]   # fixed order used everywhere
SEED = 42


def load_airline(path):
    """Load the Kaggle 'Twitter US Airline Sentiment' Tweets.csv."""
    df = pd.read_csv(path)
    df = df[["airline_sentiment", "text", "tweet_created", "airline"]].rename(
        columns={"airline_sentiment": "label"}
    )
    df = df.dropna(subset=["text", "label"])
    # Retweets / duplicates would leak between train and test.
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    return df


def make_split(df, test_size=0.2, seed=SEED):
    """Stratified train/test split performed on the RAW data (before cleaning)."""
    return train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)
