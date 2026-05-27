import re
import pickle
import warnings
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")

def download_nltk():
    for r in ["punkt", "stopwords", "wordnet", "omw-1.4"]:
        nltk.download(r, quiet=True)


def load_data():
    fake = pd.read_csv( "fake.csv")
    true = pd.read_csv( "true.csv")

    fake["label"] = 1
    true["label"] = 0

    df = pd.concat([fake, true])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(df["label"].value_counts())
    return df


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    tokens = text.split()  # fast
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def main():
    print("🚀 Training Started")

    download_nltk()
    df = load_data()

    print("🔧 Cleaning text...")
    df["content"] = df["title"] + " " + df["text"]
    df["content"] = df["content"].apply(clean_text)
    df = df[df["content"].str.strip() != ""]

    print("📐 TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    X = vectorizer.fit_transform(df["content"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("🏋️ Training Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # Evaluation
    pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("F1 Score:", f1_score(y_test, pred))

    # Save
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("💾 Model saved successfully!")

if __name__ == "__main__":
    main()