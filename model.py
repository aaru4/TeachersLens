import cv2
import pytesseract

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras import backend as K

# Set Tesseract OCR executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\aarus\coding\tesseract.exe'

# Clear the Keras session to start fresh
K.clear_session()

# Define the perform_ocr function for text extraction
def perform_ocr(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Perform OCR using pytesseract
    text = pytesseract.image_to_string(image)

    return text

# Example usage of the perform_ocr function
image_path = "path/to/your/image.png"
text_results = perform_ocr(image_path)
print(text_results)

# Load the dataset using pandas
data = pd.read_csv(r'C:\Users\aarus\coding\aiDetectApp\GPT-wiki-intro.csv')

# Extract the relevant columns
wiki_intro_data = data['wiki_intro'].values
generated_intro_data = data['generated_intro'].values

# Create labels for non-AI content (0) and AI-generated content (1)
non_ai_labels = np.zeros(len(wiki_intro_data))
ai_labels = np.ones(len(generated_intro_data))

# Combine the labels for both non-AI and AI-generated content
labels = np.concatenate((non_ai_labels, ai_labels))

# Tokenize the text data
vocab_size = 10000
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(np.concatenate([wiki_intro_data, generated_intro_data]))
wiki_intro_sequences = tokenizer.texts_to_sequences(wiki_intro_data)
generated_intro_sequences = tokenizer.texts_to_sequences(generated_intro_data)

# Pad sequences to a fixed length
max_length = 400  # Update max_length to 400
padded_wiki_intro_sequences = pad_sequences(wiki_intro_sequences, maxlen=max_length)
padded_generated_intro_sequences = pad_sequences(generated_intro_sequences, maxlen=max_length)

# Combine the padded sequences for both non-AI and AI-generated content
inputs = np.concatenate((padded_wiki_intro_sequences, padded_generated_intro_sequences))

# Create the RNN model
model = models.Sequential([
    layers.Embedding(vocab_size, 32, input_length=max_length),
    layers.SimpleRNN(64),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
model.fit(inputs, labels, epochs=5, batch_size=128, validation_split=0.1)

# Now, you can use the text_results for AI detection
# For example, you can use a simple keyword check to determine if the text contains AI-generated content
# Replace this logic with your actual AI detection approach

if "AI-generated content" in text_results:
    print("AI-generated content detected.")
else:
    print("Non-AI content detected.")
