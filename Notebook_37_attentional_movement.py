import math
import numpy as np
import tensorflow as tf
import pandas as pd
import os.path
import datetime as dt
import matplotlib.pyplot as plt
import time
import datetime as dt
from pymongo import MongoClient
from bson.binary import Binary
import pickle

#
messages = MongoClient()['telegram_migrate']['messages']
# messages.create_index([('channel', 1)])
# messages.create_index([('_date', 1)])

# TODO bidirectional
# second one don't have to send state so can be lstm :)
# TODO decoder has two gru each one can return state


class DataLoader:
    def __init__(self, fs, split, cols):
        dfs = [pd.read_csv(f) for f in fs]
        min_df_length = min([len(df) for df in dfs])
        i_split = int(min_df_length * (1. - split))
        self.data_train = [df.get(cols).values[-min_df_length:-i_split] for df in dfs]
        self.data_test = [df.get(cols).values[-i_split:] for df in dfs]
        self.data_train = np.concatenate(self.data_train, axis=1)
        self.data_test = np.concatenate(self.data_test, axis=1)
        self.time_train = dfs[0]['Date'].values[-min_df_length:- i_split]
        self.time_test = dfs[0]['Date'].values[-i_split:]
        self.len_train = len(self.data_train)
        self.len_test = len(self.data_test)
        self.len_train_windows = None

    def get_test_data(self, seq_len, normalise):
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])

        data_windows = np.array(data_windows).astype(np.float32)
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows

        raw_date = self.time_test[0].split('-')
        today = dt.datetime(year=int(raw_date[0]), month=int(raw_date[1]), day=int(raw_date[2]))
        ms = messages.aggregate(
            [{'$match': {'_date': {'$gte': today, '$lt': today + dt.timedelta(days=1)}, 'elmo': {'$exists': True}}},
             {'$sample': {'size': news_messages}}])
        ms = np.stack([pickle.loads(m['elmo']) for m in ms])
        # print(ms.shape)
        x = data_windows[:, :-1]
        y = data_windows[:, -1, [0]]
        return ms, x, y

    def get_train_data(self, seq_len, normalise):
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x).astype(np.float32), np.array(data_y).astype(np.float32)

    def generate_train_batch(self, seq_len, batch_size, normalise):
        i = 0
        while i < (self.len_train - seq_len):
            news_batch = []
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(news_batch), np.array(x_batch).astype(np.float32), np.array(y_batch).astype(np.float32)
                    i = 0
                news, x, y = self._next_window(i, seq_len, normalise)
                news_batch.append(news)
                x_batch.append(x)
                y_batch.append(y)
                i += 1

            yield np.array(news_batch), np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalise):
        window = self.data_train[i:i+seq_len]
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        raw_date = self.time_train[i + seq_len - 2].split('-')
        today = dt.datetime(year=int(raw_date[0]), month=int(raw_date[1]), day=int(raw_date[2]))
        # t0 = time.time()
        ms = messages.aggregate([{'$match': {'_date': {'$gte': today, '$lt': today + dt.timedelta(days=1)}, 'elmo': {'$exists': True}}}, {'$sample': {'size': news_messages}}])
        ms = np.stack([pickle.loads(m['elmo']) for m in ms])
        # print(time.time() - t0)
        x = window[:-1]
        y = window[-1]
        return ms, x, y

    @staticmethod
    def normalise_windows(window_data, single_window=False):
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                normalised_col = [((float(p) / float(window[0, col_i])) - 1) if window[0, col_i] != 0 else 0 for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T  # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data).astype(np.float32)


# model = tf.keras.Sequential([
#     tf.keras.layers.TimeDistributed(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
#     tf.keras.layers.Dense(64),
#     tf.keras.layers.LeakyReLU(alpha=0.05),
#     tf.keras.layers.Dense(1)
# ])


class Encoder(tf.keras.Model):
    def __init__(self, units, batch_sz):
        super(Encoder, self).__init__()

        self.units = units
        self.batch_sz = batch_sz

        self.m_rcc = tf.keras.layers.TimeDistributed(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units, recurrent_initializer='glorot_uniform')))
        self.rcc = tf.keras.layers.GRU(units, recurrent_initializer='glorot_uniform', return_state=True)  # TODO cant make it bidirectional
        # self.h_dense = tf.keras.layers.Dense(units),
        # self.activation = tf.keras.layers.LeakyReLU(alpha=0.1),
        # self.dense = tf.keras.layers.Dense(1)

    def call(self, x, hidden):
        x = self.m_rcc(x)
        x, state = self.rcc(x, initial_state=hidden)
        # x = self.drop_1(x)
        return x, state

    def initialize_hidden_state(self, batch_sz=None):
        return tf.zeros((batch_sz if batch_sz else self.batch_sz, self.units))


news_vector_size = 64  # 64
news_messages = 32  # 73
word_vector_size = 200  # 200

batch_size = 16
r_hidden = 256

encoder = Encoder(r_hidden, batch_sz=batch_size)
sample_state = encoder.initialize_hidden_state()

example_input_batch = tf.random.uniform((batch_size, news_messages, news_vector_size, word_vector_size), minval=0, maxval=1)
sample_output, sample_state = encoder(example_input_batch, sample_state)
print(sample_output)


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


attention_layer = BahdanauAttention(10)
attention_result, attention_weights = attention_layer(sample_state, sample_output)


# model = tf.keras.models.Sequential([
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(r_hidden, input_shape=(window - 1, len(symbols) * len(csv_columns)), return_sequences=True)),
#     # tf.keras.layers.LeakyReLU(.1),
#     tf.keras.layers.Dropout(.2),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(r_hidden, return_sequences=True)),
#     # tf.keras.layers.LeakyReLU(.1),
#     tf.keras.layers.LSTM(r_hidden),
#     # tf.keras.layers.LeakyReLU(.05),
#     tf.keras.layers.Dropout(.2),
#     tf.keras.layers.Dense(len(symbols) * len(csv_columns), activation='linear')
# ])


class Decoder(tf.keras.Model):
    def __init__(self, window, symbols, csv_columns, units, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.window = window
        self.symbols = symbols
        self.csv_columns = csv_columns
        self.dec_units = dec_units
        self.units = units

        self.rcc_0 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, input_shape=(window - 1, len(symbols) * len(csv_columns)),
                                         return_state=False, return_sequences=True,
                                         recurrent_initializer='glorot_uniform'))
        self.drop_0 = tf.keras.layers.Dropout(.2)
        self.rcc_1 = tf.keras.layers.GRU(units, return_state=True, return_sequences=False,
                                         recurrent_initializer='glorot_uniform')
        self.drop_1 = tf.keras.layers.Dropout(.2)
        self.dense = tf.keras.layers.Dense(len(symbols) * len(csv_columns), activation='linear')

        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        x = self.rcc_0(x)
        x = self.drop_0(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(tf.concat([context_vector, context_vector], 1), 1), x], axis=1)

        # passing the concatenated vector to the GRU
        output, state = self.rcc_1(x)

        # output shape == (batch_size * 1, hidden_size)
        # print(output.shape)
        # output = tf.reshape(output, (-1, output.shape[1]))
        # print(output.shape)
        # output shape == (batch_size, vocab)
        x = self.dense(output)

        # # x, state_0 = self.rcc_0(x)
        # # x = self.drop_0(x)
        # # x, state_1 = self.rcc_1(x, initial_state=hidden)
        # x = self.drop_1(x)
        # output = self.dense(x)
        return x, state, attention_weights


# class Encoder(tf.keras.Model):
#     def __init__(self, window, symbols, csv_columns, units, batch_sz):
#         super(Encoder, self).__init__()
#         self.window = window
#         self.symbols = symbols
#         self.csv_columns = csv_columns
#         self.units = units
#         self.batch_sz = batch_sz
#
#         self.rcc_0 = tf.keras.layers.GRU(units, input_shape=(window - 1, len(symbols) * len(csv_columns)), return_state=True, return_sequences=True, recurrent_initializer='glorot_uniform')
#         self.drop_0 = tf.keras.layers.Dropout(.2)
#         self.rcc_1 = tf.keras.layers.GRU(units, return_state=True, return_sequences=False, recurrent_initializer='glorot_uniform')
#         self.drop_1 = tf.keras.layers.Dropout(.2)
#         self.dense = tf.keras.layers.Dense(len(symbols) * len(csv_columns), activation='linear')
#
#     def call(self, x, hidden_0, hidden_1):
#         x, state_0 = self.rcc_0(x, initial_state=hidden_0)
#         x = self.drop_0(x)
#         x, state_1 = self.rcc_1(x, initial_state=hidden_1)
#         x = self.drop_1(x)
#         output = self.dense(x)
#         return output, state_0, state_1
#
#     def initialize_hidden_state(self):
#         return tf.zeros((self.batch_sz, self.units)), tf.zeros((self.batch_sz, self.units))
#         # return tf.zeros((self.units, self.units)), tf.zeros((self.units, self.units))


_window = 32
date = '2020-05-07'
_symbols = ['BMLT1', 'BSDR1', 'BTEJ1', 'GCOZ1', 'IKHR1', 'IPTR1', 'MSMI1', 'PNBA1', 'PNES1', 'PTEH1']
_symbols = [os.path.join('Symbols', f"{symbol}_{date}.csv") for symbol in _symbols]
_csv_columns = ["Open", "Close", "Volume", "Low", "High"]

data = DataLoader(_symbols, split=.6, cols=_csv_columns)

# encoder = Encoder(_window, _symbols, _csv_columns, r_hidden, batch_sz=batch_size)
# sample_state_0, sample_state_1 = encoder.initialize_hidden_state()
#
example_input_batch = tf.random.uniform((batch_size, _window - 1, len(_symbols) * len(_csv_columns)), minval=0, maxval=1)
# sample_output, sample_state_0, sample_state_1 = encoder(example_input_batch, sample_state_0, sample_state_1)
# print(sample_output)

#  --------------


decoder = Decoder(_window, _symbols, _csv_columns, r_hidden, r_hidden, batch_sz=batch_size)
sample_decoder_output, _, _ = decoder(example_input_batch, sample_state, sample_output)
print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

optimizer = tf.keras.optimizers.Adam(1e-4)
loss_object = tf.keras.losses.MeanSquaredError()

checkpoint_dir = './attention_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)


@tf.function
def train(news, x, y, news_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        news_output, news_hidden = encoder(news, news_hidden)
        x_hidden = news_hidden
        # dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)
        # dec_input = tf.expand_dims([0] * batch_size, 1)
        predictions, news_hidden, _ = decoder(x, x_hidden, news_output)

        loss += loss_object(y, predictions)
    batch_loss = (loss / int(x.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss


EPOCHS = 24
# steps_per_epoch = 398 // batch_size


for epoch in range(EPOCHS):
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    # for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
    # _0 = tf.random.uniform((64, 2), minval=0, maxval=vocab_inp_size - 1, dtype=tf.int32)
    # _news = tf.random.uniform((batch_size, news_messages, news_vector_size, word_vector_size), minval=0, maxval=1)
    # _x = tf.random.uniform((batch_size, _window - 1, len(_symbols) * len(_csv_columns)), minval=0, maxval=1)
    # _y = tf.random.uniform((batch_size, len(_symbols) * len(_csv_columns)), minval=0, maxval=1)
    steps_per_epoch = 398 // batch_size
    steps = 0
    for (batch, (news, x, y)) in enumerate(data.generate_train_batch(
        seq_len=_window,
        batch_size=batch_size,
        normalise=True
    )):
        if steps > steps_per_epoch:
            break
        else:
            steps += 1
        while x.shape[0] != batch_size:
            filler_size = batch_size - x.shape[0]
            news = tf.concat([news, news[-filler_size:]], 0)
            x = tf.concat([x, x[-filler_size:]], 0)
            y = tf.concat([y, y[-filler_size:]], 0)
        # news = _news
        batch_loss = train(news, x, y, enc_hidden)
        total_loss += batch_loss

        # if batch % 100 == 0:
        #     print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))
    # print(total_loss)
    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

    # print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
    # print(batch)
    print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

#
# _news = tf.random.uniform((1, news_messages, news_vector_size, word_vector_size), minval=0, maxval=1)
# print(_news.shape)


def predict_sequences_multiple(_news, data, window_size, prediction_len):
    news_hidden = encoder.initialize_hidden_state(batch_sz=1)
    # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    print('[Model] Predicting Sequences Multiple...')
    prediction_seqs = []
    for i in range(int(len(data) / prediction_len)):
        curr_frame = data[i * prediction_len]
        predicted = []
        for j in range(prediction_len):

            news_output, news_hidden = encoder(_news, news_hidden)
            x_hidden = news_hidden
            predictions, news_hidden, _ = decoder(curr_frame[np.newaxis, :, :], x_hidden, news_output)

            predicted.append(predictions[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


news_test, x_test, y_test = data.get_test_data(seq_len=_window, normalise=True)
predictions = predict_sequences_multiple(news_test[np.newaxis], x_test, _window, _window)
plot_results_multiple(predictions, y_test, _window)
