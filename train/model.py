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
from tensorflow.keras.regularizers import l2

import re
import os
import json
from contractions import fix
import matplotlib.pyplot as plt



# Load pre-trained embedding
from util import load_legacy_w2v


def split_and_save_data(X, Y, data, output_dir, test_size=0.3, val_size=0.15):
    """
    Split data into train, validation, and test sets and save as TSV files.
    Args:
        X: Input features.
        Y: Labels.
        data: Original dataframe with all columns.
        output_dir: Directory to save splits.
        test_size: Fraction of data to use for testing.
        val_size: Fraction of training data to use for validation.
    Returns:
        Split datasets: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Check if splits already exist
    train_file = os.path.join(output_dir, 'train.tsv')
    val_file = os.path.join(output_dir, 'val.tsv')
    test_file = os.path.join(output_dir, 'test.tsv')

    if os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file):
        print("Train, validation, and test splits already exist. Loading splits.")
        train_data = pd.read_csv(train_file, sep='\t')
        val_data = pd.read_csv(val_file, sep='\t')
        test_data = pd.read_csv(test_file, sep='\t')

        train_data['embed'] = train_data['embed'].apply(json.loads)
        val_data['embed'] = val_data['embed'].apply(json.loads)
        test_data['embed'] = test_data['embed'].apply(json.loads)


        return (
            np.array(train_data['embed'].tolist()),
            np.array(val_data['embed'].tolist()),
            np.array(test_data['embed'].tolist()),
            train_data['label'], val_data['label'], test_data['label']
        )

    print("Creating new train, validation, and test splits.")

    if 'embed' in data.columns:
        print("Warning: Overwriting existing 'embed' column in data.")
    data['embed'] = [json.dumps([int(element) for element in row]) for row in X]


    # Split into train and test
    X_train, X_temp, y_train, y_temp, data_train, data_temp = train_test_split(
        X, Y, data, test_size=test_size, random_state=42
    )
    # Split temp into validation and test
    X_val, X_test, y_val, y_test, data_val, data_test = train_test_split(
        X_temp, y_temp, data_temp, test_size=0.5, random_state=42
    )



    # Save splits
    data_train.to_csv(train_file, sep='\t', index=False)
    data_val.to_csv(val_file, sep='\t', index=False)
    data_test.to_csv(test_file, sep='\t', index=False)

    return X_train, X_val, X_test, y_train, y_val, y_test


def plot_training_history(history, output_path):
    """
    Plots training and validation loss over epochs.
    Args:
        history: Keras History object.
        output_path: Path to save the plot image.
    """
    # Extract loss values
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(train_loss) + 1)

    # Create the plot
    plt.figure()
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Save the plot
    plt.savefig(output_path)
    plt.close()  # Close the plot to avoid display during batch runs


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

    # Create train, validation, and test splits
    output_dir = os.path.dirname(csv_file)
    X_train, X_val, X_test, y_train, y_val, y_test = split_and_save_data(
        X_padded, Y, data, output_dir
    )
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
        Dense(1, activation='sigmoid',kernel_regularizer=l2(0.01))
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

    history = model.fit(
        # X_train[:100], y_train[:100],
        X_train, y_train,
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

    # Plot training history
    plot_training_history(history, f'models/{emod}_training_loss_plot.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with pre-trained embeddings")
    parser.add_argument("--embedder_path", type=str, required=True, help="Path to the pre-trained embedder")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the CSV dataset")
    parser.add_argument("--emod", type=str, required=True, help="Embedding type")

    args = parser.parse_args()
    main(args.embedder_path, args.csv_file, args.emod)
