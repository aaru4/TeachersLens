import cv2
import pytesseract

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.graphics.texture import Texture

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras import backend as K
import threading

# Set Tesseract OCR executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\aarus\coding\tesseract.exe'

class CameraApp(App):
    def __init__(self, **kwargs):
        super(CameraApp, self).__init__(**kwargs)
        self.tokenizer = None  # Initialize tokenizer to None
        self.ai_model = None
        self.load_tokenizer()

    def load_tokenizer(self):
        # Load the dataset using pandas
        data = pd.read_csv(r'C:\Users\aarus\coding\aiDetectApp\ai_detector\GPT-wiki-intro.csv')

        # Extract the relevant columns
        self.wiki_intro_data = data['wiki_intro'].values
        self.generated_intro_data = data['generated_intro'].values

        # Create labels for non-AI content (0) and AI-generated content (1)
        non_ai_labels = np.zeros(len(self.wiki_intro_data))
        ai_labels = np.ones(len(self.generated_intro_data))

        # Combine the labels for both non-AI and AI-generated content
        labels = np.concatenate((non_ai_labels, ai_labels))

        # Tokenize the text data
        vocab_size = 10000
        tokenizer = Tokenizer(num_words=vocab_size)
        tokenizer.fit_on_texts(np.concatenate([self.wiki_intro_data, self.generated_intro_data]))

        self.tokenizer = tokenizer

        # Pad sequences to a fixed length
        max_length = 400  # Update max_length to 400
        padded_wiki_intro_sequences = pad_sequences(tokenizer.texts_to_sequences(self.wiki_intro_data), maxlen=max_length)
        padded_generated_intro_sequences = pad_sequences(tokenizer.texts_to_sequences(self.generated_intro_data), maxlen=max_length)

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

        self.ai_model = model

    def build(self):
        self.capture = cv2.VideoCapture(0)
        layout = BoxLayout(orientation='vertical')

        # Create a button to take pictures
        self.capture_button = Button(text='Take Picture', size_hint=(1, 0.2))
        self.capture_button.bind(on_press=self.on_capture)
        layout.add_widget(self.capture_button)

        # Create an image display
        self.image = Image()
        layout.add_widget(self.image)

        # Create a label to display OCR-detected text
        self.ocr_text_label = Label(text='', size_hint=(1, 0.2), halign='center', valign='middle')
        layout.add_widget(self.ocr_text_label)

        return layout

    def load_ai_model(self):
        # Load the dataset using pandas
        data = pd.read_csv(r'C:\Users\aarus\coding\aiDetectApp\ai_detector\GPT-wiki-intro.csv')

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

        return model

    def load_tokenizer_thread(self):
        # Load the tokenizer
        self.tokenizer = Tokenizer(num_words=vocab_size)
        self.tokenizer.fit_on_texts(np.concatenate([wiki_intro_data, generated_intro_data]))

    def perform_ai_detection(self, text_results):
        # Tokenize the text from OCR
        text_sequence = self.tokenizer.texts_to_sequences([text_results])
        # Pad the sequence to the same length as the AI model input
        text_padded = pad_sequences(text_sequence, maxlen=400)
        # Perform AI detection
        prediction = self.ai_model.predict(text_padded)
        # Output the result
        if prediction[0][0] >= 0.5:
            print("AI-generated content detected.")
        else:
            print("Non-AI content detected.")

    def on_capture(self, *args):
        # Capture the image from the camera
        ret, frame = self.capture.read()
        if ret:
            # Convert the image to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform OCR using pytesseract on the image data
            text_results = pytesseract.image_to_string(frame_rgb)

            # Display the image in the app UI
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
            texture.blit_buffer(frame_rgb.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
            self.image.texture = texture

            # Load the tokenizer using threading if it's not already loaded
            if self.tokenizer is None:
                threading.Thread(target=self.load_tokenizer_thread).start()

            # Load the AI model only when it's required
            if self.ai_model is None and self.tokenizer is not None:
                self.ai_model = self.load_ai_model()

            self.perform_ai_detection(text_results)

    def on_stop(self):
        # Release the camera on app exit
        self.capture.release()

if __name__ == '__main__':
    CameraApp().run()
