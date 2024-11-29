import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
import numpy as np

# Example data (English to Spanish)
english_sentences = ["hello", "how are you"]
spanish_sentences = ["hola", "cómo estás"]

# Vocabulary
eng_vocab = {word: i + 1 for i, word in enumerate(set(" ".join(english_sentences).split()))}
spa_vocab = {word: i + 1 for i, word in enumerate(set(" ".join(spanish_sentences).split()))}
reverse_spa_vocab = {i: word for word, i in spa_vocab.items()}

# Preprocess
X = [[eng_vocab[word] for word in sentence.split()] for sentence in english_sentences]
y = [[spa_vocab[word] for word in sentence.split()] for sentence in spanish_sentences]

X = tf.keras.preprocessing.sequence.pad_sequences(X, padding="post")
y = tf.keras.preprocessing.sequence.pad_sequences(y, padding="post")

# Encoder-Decoder Model for Training
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(input_dim=len(eng_vocab) + 1, output_dim=16)
encoder_embedded = encoder_embedding(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(32, return_state=True)(encoder_embedded)

decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(input_dim=len(spa_vocab) + 1, output_dim=16)
decoder_embedded = decoder_embedding(decoder_inputs)
decoder_lstm = LSTM(32, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedded, initial_state=[state_h, state_c])
decoder_dense = Dense(len(spa_vocab) + 1, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

# Training Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Prepare decoder inputs
decoder_input_data = np.zeros_like(y)
decoder_input_data[:, 1:] = y[:, :-1]  # Shift right
decoder_input_data[:, 0] = 0  # Start token
y = np.expand_dims(y, axis=-1)

# Train the model
model.fit([X, decoder_input_data], y, epochs=100, batch_size=2, verbose=2)

# Inference Models
# Encoder Model
encoder_model = Model(encoder_inputs, [state_h, state_c])

# Decoder Model
decoder_state_input_h = Input(shape=(32,))
decoder_state_input_c = Input(shape=(32,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# Reuse the embedding and LSTM layers
decoder_embedded2 = decoder_embedding(decoder_inputs)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(
    decoder_embedded2, initial_state=decoder_states_inputs
)
decoder_outputs2 = decoder_dense(decoder_outputs2)
decoder_states = [state_h2, state_c2]

decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs2] + decoder_states)

# Translate function
def translate(input_sentence):
    # Encode input sentence
    input_tokens = [[eng_vocab.get(word, 0) for word in input_sentence.split()]]
    input_tokens = tf.keras.preprocessing.sequence.pad_sequences(input_tokens, maxlen=X.shape[1], padding="post")
    states = encoder_model.predict(input_tokens)

    # Decode step-by-step
    target_seq = np.zeros((1, 1))  # Start token
    translated_sentence = []
    for _ in range(10):  # Max translation length
        output_tokens, h, c = decoder_model.predict([target_seq] + states)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_spa_vocab.get(sampled_token_index, "")
        if sampled_word == "<end>" or not sampled_word:
            break
        translated_sentence.append(sampled_word)

        # Update target sequence and states
        target_seq = np.array([[sampled_token_index]])
        states = [h, c]

    return " ".join(translated_sentence)

# Example Translation
translated_text = translate("hello")
print("Translated Text:", translated_text)
