from prefect import task, flow
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

@task
def load_data(file_path):
    """
    Load data from a CSV file.
    """
    return pd.read_csv(file_path)

@task
def split_inputs_output(data, inputs, output):
    """
    Split features and target variables.
    """
    X = data[inputs]
    y = data[output]
    return X, y

@task
def split_train_test(X, y, test_size=0.25, random_state=0):
    """
    Split data into train and test sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

@task
def preprocess_data(X_train, X_test, y_train, y_test):
    """
    Rescale the data.
    """
    def clean_text(text):
        # Remove "READ MORE"
        text = re.sub(r'READ MORE', '', text)

        # Remove numbers
        text = re.sub(r'\d+', '', text)

        # Remove punctuation marks
        text = re.sub(f"[{string.punctuation}]", "", text)

        # Convert text to lowercase
        text = text.lower()

        # Tokenize the text into words
        tokens = word_tokenize(text)

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]

        # Lemmatize words to their base form
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        # Join tokens back into a single string
        cleaned_text = ' '.join(tokens)

        return cleaned_text

    for col in X_train.columns:
        X_train[col] = X_train[col].apply(lambda doc: clean_text(doc))
    for col in X_test.columns:
        X_test[col] = X_test[col].apply(lambda doc: clean_text(doc))
    
    return X_train, X_test, y_train, y_test

@task
def train_model(X_train, y_train, hyperparameters):
    """
    Training the machine learning model.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('text_1', TfidfVectorizer(), 'Review'),  # TF-IDF vectorization for text_data_1   
        ],
        remainder='passthrough'  # Keep other columns unchanged
    )
    clf = Pipeline(
        [
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(**hyperparameters))
        ]
    )
    clf.fit(X_train, y_train)
    return clf

@task
def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluating the model.
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_score = metrics.f1_score(y_train, y_train_pred)
    test_score = metrics.f1_score(y_test, y_test_pred)
    
    return train_score, test_score

@flow(name="LR Training Flow")
def workflow():
    DATA_PATH = "sentiment_data.csv"
    INPUTS = ['Review']
    OUTPUT = 'Review_Type'
    HYPERPARAMETERS = {'C': 1, 'penalty': "l2"}
    
    # Load data
    data = load_data(DATA_PATH)
    
    # Identify Inputs and Output
    X, y = split_inputs_output(data, INPUTS, OUTPUT)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(X_train, X_test, y_train, y_test)

    # Build a model
    model = train_model(X_train, y_train, HYPERPARAMETERS)
    
    # Evaluation
    train_score, test_score = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    print("Train Score:", train_score)
    print("Test Score:", test_score)

if __name__ == "__main__":
    workflow.serve(
        name="my-first-deployment",
        cron="5 4 2 * *"
    )