import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Load JSON dataset
with open("dataset.json", "r", encoding="utf-8") as f:
    datastore = json.load(f)

# Initialize lists
sentences = []
categories = []

# Load texts and labels
for item in datastore:
    sentences.append(item['sentence'])
    categories.append(item['category'])

print(f"Loaded {len(sentences)} samples")

unique_categories = list(set(categories))
print(f"Unique categories: {unique_categories}")

# Encode labels into numerical values
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(categories)  

# Check encoding
print(f"Example category mapping: {dict(zip(unique_categories, label_encoder.transform(unique_categories)))}")

# Tokenization parameters
vocab_size = 10000
max_length = 32
embedding_dim = 16
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"

# Initialize tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(sentences)

# Convert texts into numerical sequences
sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Convert labels to a numpy array
encoded_labels = np.array(encoded_labels, dtype=np.int32)

# Split data into training and testing sets 
training_size = int(len(sentences) * 0.8)  # 80% for training, 20% for testing
training_sentences = padded_sequences[:training_size]
testing_sentences = padded_sequences[training_size:]

training_labels = encoded_labels[:training_size]
testing_labels = encoded_labels[training_size:]

print(f"Training size: {len(training_sentences)}, Testing size: {len(testing_sentences)}")

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
    tf.keras.layers.Dense(48, activation='relu'),
    tf.keras.layers.Dropout(0.3),  # Adding Dropout for regularization
    tf.keras.layers.Dense(len(unique_categories), activation='softmax')  # Multi-class classification
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=8,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0
)

# Print model summary
print(model.summary())

# Train the model 
num_epochs = 30
history = model.fit(training_sentences, training_labels, epochs=num_epochs, validation_data=(testing_sentences, testing_labels), verbose=2)

print("Training complete!")

# Function to predict the category of an animal 
def predict_animal(sentence):
    sequence = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(sequence, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    prediction = model.predict(padded)
    predicted_label = np.argmax(prediction)
    return label_encoder.inverse_transform([predicted_label])[0]

# Example prediction
test_sentence = "There might be some cat in the picture, am I right?"
print(f"Predicted category: {predict_animal(test_sentence)}")

# Save the model in Keras format
model.save("animal_classifier_nlp.keras")

# Save the tokenizer
import json
tokenizer_json = tokenizer.to_json()
with open("tokenizer.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))

print("Model and tokenizer saved!")
