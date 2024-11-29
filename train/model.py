import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, GRU
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import re
from contractions import fix


# Load pre-trained embedding
from util import load_legacy_w2v

# Define F1 metric callback
class F1Metric(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        y_pred = (self.model.predict(X_val) > 0.5).astype(int)
        macro_f1 = f1_score(y_val, y_pred, average='macro')
        print(f"Epoch {epoch + 1}: Macro-F1 Score = {macro_f1:.4f}")

def preprocess_text(text):
    """
    Preprocess text to ensure alignment with embedding vocabulary.
    Handles contractions, converts to lowercase, and removes unnecessary characters.
    """
    # Expand contractions
    text = fix(text)

    # Convert to lowercase
    text = text.lower()

    # Remove special characters but keep basic punctuation
    text = re.sub(r"[^\w\s]", '', text)  # Removes the apostrophe
    return text


def align_tokenizer_with_embeddings(X, embedder, max_vocab_size, max_length, dim):
    """
    Tokenize text and align tokenizer vocabulary with the pretrained embeddings.
    """
    # Preprocess text
    X = X.apply(preprocess_text)

    # Initialize tokenizer
    tokenizer = Tokenizer(num_words=max_vocab_size)
    tokenizer.fit_on_texts(X)

    # Convert text to sequences
    X_seq = tokenizer.texts_to_sequences(X)
    X_padded = pad_sequences(X_seq, maxlen=max_length, padding='post')

    # Convert embedding dictionary to lowercase keys
    lowercase_embedder = {key.lower(): value for key, value in embedder.items()}


    # Build embedding matrix
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((min(max_vocab_size, len(word_index) + 1), dim))

    missing = []
    for word, i in word_index.items():
        if i >= max_vocab_size:
            continue
        # Retrieve embedding vector in a case-insensitive manner
        embedding_vector = lowercase_embedder.get(word.lower())
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            missing.append(word)

    # Log missing embeddings
    print(f"Number of missing embeddings: {len(missing)}")
    if len(missing) > 0:
        print(f"Sample missing words: {missing[:10]}")  # Inspect a few missing words for patterns

    return X_padded, tokenizer, embedding_matrix

def main(embedder_path, csv_file, emod):
    # Load pre-trained embedder
    embedder, dim = load_legacy_w2v(embedder_path, dim=300)

    # Load dataset
    data = pd.read_csv(csv_file, sep='\t')

    # Extract tokenised_text and label
    X = data['text']
    Y = data['label']

    # Tokenize the text
    max_vocab_size = 10000
    max_length = 40  # Adjust this as needed
    X_padded, tokenizer, embedding_matrix = align_tokenizer_with_embeddings(X, embedder, max_vocab_size, max_length, dim)

    # Split into train-validation-test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X_padded, Y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Define the embedding layer
    embedding_layer = Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=dim,
        weights=[embedding_matrix],
        input_length=max_length,
        trainable=False  # Freeze weights to prevent training
    )

    # Build the model
    model = Sequential([
        embedding_layer,
        Bidirectional(GRU(units=200, kernel_initializer="glorot_uniform")),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    checkpoint_callback = ModelCheckpoint(
        filepath='models/'+emod+'_best_model.h5',  # Path to save the model
        monitor='val_loss',              # Metric to monitor
        save_best_only=True,             # Save only when performance improves
        mode='min',                      # Minimize val_loss
        verbose=1                        # Print save message
    )

    f1_callback = F1Metric(validation_data=(X_val, y_val))

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))

    model.fit(
        X_train[:100], y_train[:100],
        # X_train, y_train,
        batch_size=64,
        epochs=10,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=[checkpoint_callback, f1_callback]
    )

    # Evaluate the model
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Macro-F1 Score: {macro_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with pre-trained embeddings")
    parser.add_argument("--embedder_path", type=str, required=True, help="Path to the pre-trained embedder")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the CSV dataset")
    parser.add_argument("--emod", type=str, required=True, help="Embedding type")

    args = parser.parse_args()
    main(args.embedder_path, args.csv_file, args.emod)