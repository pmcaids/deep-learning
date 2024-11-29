import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
import numpy as np

# Example data
sentences = ["I love coding", "AI is amazing"]
tags = [["PRP", "VBP", "VBG"], ["NNP", "VBZ", "JJ"]]

'''PRP: Personal Pronoun
VBP: Verb, Present Tense
VBG: Verb, Gerund/Present Participle
NNP: Proper Noun
VBZ: Verb, 3rd Person Singular Present
JJ: Adjective'''

# Prepare vocabulary and tag mappings
word_vocab = {word: i + 1 for i, word in enumerate(set(" ".join(sentences).split()))}
tag_vocab = {tag: i + 1 for i, tag in enumerate(set(tag for tag_seq in tags for tag in tag_seq))}
reverse_tag_vocab = {i: tag for tag, i in tag_vocab.items()}

# Convert sentences and tags to integers
X = [[word_vocab[word] for word in sentence.split()] for sentence in sentences]
y = [[tag_vocab[tag] for tag in tag_seq] for tag_seq in tags]

# Pad sequences
X = tf.keras.preprocessing.sequence.pad_sequences(X, padding="post")
y = tf.keras.preprocessing.sequence.pad_sequences(y, padding="post")

# Model architecture
input_ = Input(shape=(None,))
x = Embedding(input_dim=len(word_vocab) + 1, output_dim=16)(input_)
x = LSTM(32, return_sequences=True)(x)
output = Dense(len(tag_vocab) + 1, activation="softmax")(x)

model = Model(input_, output)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
y = np.expand_dims(y, axis=-1)  # Expand dimensions for sparse_categorical_crossentropy
model.fit(X, y, epochs=10, batch_size=2)

# Predict on new data
test_sentence = "I am learning"
test_X = [[word_vocab.get(word, 0) for word in test_sentence.split()]]
test_X = tf.keras.preprocessing.sequence.pad_sequences(test_X, maxlen=X.shape[1], padding="post")
predictions = model.predict(test_X)

# Decode predictions
pred_tags = [reverse_tag_vocab[np.argmax(p)] for p in predictions[0]]
print(pred_tags)
