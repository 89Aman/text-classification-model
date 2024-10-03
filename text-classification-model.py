import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from joblib import parallel_backend
import argparse


# Function to load data in chunks
def load_data_in_chunks(file_path, chunk_size=20000):
    X_list = []
    y_list = []

    for chunk in pd.read_csv(file_path, chunksize=chunk_size, encoding='ISO-8859-1'):
        X_list.extend(chunk['text'].tolist())
        if 'target' in chunk.columns:
            y_list.extend(chunk['target'].tolist())

    return X_list, y_list


# Function to train the model
def train_model(X_train, y_train, model_type='svm'):
    if model_type == 'svm':
        print("Training LinearSVC model with parallel processing...")
        model = LinearSVC()
    else:
        print("Training Naive Bayes model...")
        model = MultinomialNB()

    # Train the model using all available CPU cores
    with parallel_backend('threading', n_jobs=-1):
        model.fit(X_train, y_train)

    return model


# Function to save the predictions
def save_predictions(output_file, test_data, predicted_categories):
    output_data = pd.DataFrame({
        'Index': test_data['Index'],
        'target': predicted_categories
    })
    output_data.to_csv(output_file, index=False)
    print(f"Predictions saved to '{output_file}'")


# Main function to load data, vectorize, train, and predict
def main(train_file, test_file, output_file, model_type='svm'):
    chunk_size = 20000

    # Step 1: Load the training data in chunks
    print(f"Loading and processing the training data from {train_file} in chunks...")
    X_train_list, y_train_list = load_data_in_chunks(train_file, chunk_size)

    # Step 2: Vectorize the training data using HashingVectorizer
    print("Vectorizing the training data using HashingVectorizer...")
    vectorizer = HashingVectorizer(n_features=5000, stop_words='english')
    X_train_hashed = vectorizer.fit_transform(X_train_list)

    # Step 3: Train the model
    model = train_model(X_train_hashed, y_train_list, model_type)

    # Step 4: Load and vectorize the test data
    print("Loading and vectorizing the test data...")
    test_data = pd.read_csv(test_file, encoding='ISO-8859-1')
    X_test = test_data['text']
    X_test_hashed = vectorizer.transform(X_test)

    # Step 5: Make predictions
    print("Making predictions on the test data...")
    predicted_categories = model.predict(X_test_hashed)

    # Step 6: Save predictions to CSV
    save_predictions(output_file, test_data, predicted_categories)


# Command-line argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test an ML model on large datasets")
    parser.add_argument('--train_file', type=str, required=True, help="Path to the training CSV file")
    parser.add_argument('--test_file', type=str, required=True, help="Path to the test CSV file")
    parser.add_argument('--output_file', type=str, default='predicted_test_data.csv',
                        help="Path to the output predictions file")
    parser.add_argument('--model', type=str, default='svm', choices=['svm', 'nb'],
                        help="Model type: 'svm' for LinearSVC or 'nb' for Naive Bayes")

    args = parser.parse_args()
    main(args.train_file, args.test_file, args.output_file, args.model)
