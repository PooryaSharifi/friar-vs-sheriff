import math
import numpy as np
import tensorflow as tf
import pandas as pd
import os.path
import datetime as dt
import matplotlib.pyplot as plt

# TODO min max
# TODO add attention + news


class DataLoader:
    def __init__(self, fs, split, cols, overlap):
        dfs = [pd.read_csv(f) for f in fs]
        min_df_length = min([len(df) for df in dfs])
        # i_split = int(min_df_length * split)
        i_split = int(min_df_length * (1. - split))
        self.data_train = [df.get(cols).values[-min_df_length:-i_split] for df in dfs]
        self.data_test = [df.get(cols).values[-i_split - overlap:] for df in dfs]
        self.data_train = np.concatenate(self.data_train, axis=1)
        self.data_test = np.concatenate(self.data_test, axis=1)
        self.time_train = dfs[0]['Date'].values[-min_df_length:-i_split]
        self.time_test = dfs[0]['Date'].values[-i_split - overlap:]
        self.len_train = len(self.data_train)
        self.len_test = len(self.data_test)

    def get_test_data(self, seq_len, normalise):
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])

        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows

        x = data_windows[:, :-1]
        y = data_windows[:, -1, [0]]
        return x, y

    def get_train_data(self, seq_len, normalise):
        data_news = []
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            new, x, y = self._next_window(i, seq_len, normalise)
            data_news.append(new)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_news), np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalise):
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalise):
        window = self.data_train[i:i + seq_len]
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        elmo = f'news/{self.time_train[i + seq_len - 2]}.npy'
        elmo = np.load(elmo)
        x = window[:-1]
        y = window[-1]
        return elmo, x, y

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
        return np.array(normalised_data)


window = 64
# symbols = [os.path.join('Symbols', "AMZN.csv"), os.path.join('Symbols', "GE.csv")]
date = '2020-05-07'
symbols = ['BMLT1', 'BSDR1', 'BTEJ1', 'GCOZ1', 'IKHR1', 'IPTR1', 'MSMI1', 'PNBA1', 'PNES1', 'PTEH1']
symbols = [os.path.join('Symbols', f"{symbol}_{date}.csv") for symbol in symbols]
csv_columns = ["Open", "Close", "Volume", "Low", "High"]

r_hidden = 128

# model = tf.keras.models.Sequential([
#     tf.keras.Input((window - 1, len(symbols) * len(csv_columns))),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(r_hidden, return_sequences=True)),
#     # tf.keras.layers.LeakyReLU(.1),
#     tf.keras.layers.Dropout(.2),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(r_hidden, return_sequences=True)),
#     # tf.keras.layers.LeakyReLU(.1),
#     tf.keras.layers.LSTM(r_hidden),
#     # tf.keras.layers.LeakyReLU(.05),
#     tf.keras.layers.Dropout(.2),
#     tf.keras.layers.Dense(len(symbols) * len(csv_columns), activation='linear')
# ])
#
# model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(1e-4))
#
# model.summary()

news_r_hidden = 64
news_messages = 64
news_vector_size = 64
word_vector_size = 200


news_input = tf.keras.layers.Input((news_messages, news_vector_size, word_vector_size))
news_flow = tf.keras.layers.TimeDistributed(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(news_r_hidden)))(news_input)
news_flow = tf.keras.layers.LeakyReLU(.1)(news_flow)
news_flow, f_state_h, f_state_c, b_state_h, b_state_c = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(r_hidden, return_state=True))(news_flow)
news_flow = tf.keras.layers.LeakyReLU(.1)(news_flow)
# news_flow, state_h, state_c = tf.keras.layers.LSTM(news_r_hidden, )(news_flow)
encoder_states = [f_state_h, f_state_c, b_state_h, b_state_c]

# news_flow = tf.keras.layers.LeakyReLU(alpha=0.05)(news_flow)
# news_flow = tf.keras.layers.Dense(1)(news_flow)
# news_model = tf.keras.Model(news_input, news_flow)
#
# news_model.summary()

trade_input = tf.keras.Input((window - 1, len(symbols) * len(csv_columns)))
trade_flow = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(r_hidden, return_sequences=True))(trade_input, initial_state=encoder_states)
trade_flow = tf.keras.layers.Dropout(.2)(trade_flow)
trade_flow = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(r_hidden, return_sequences=True))(trade_flow, initial_state=encoder_states)
trade_flow = tf.keras.layers.LSTM(r_hidden)(trade_flow)
trade_flow = tf.keras.layers.Dropout(.2)(trade_flow)

trade_flow = tf.keras.layers.concatenate([news_flow, trade_flow])
trade_flow = tf.keras.layers.Dense(len(symbols) * len(csv_columns) * 2)(trade_flow)
trade_flow = tf.keras.layers.Dense(len(symbols) * len(csv_columns), activation='linear')(trade_flow)

model = tf.keras.Model([news_input, trade_input], trade_flow)
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(1e-4))

model.summary()


def train_generator(data_gen, epochs, batch_size, steps_per_epoch, save_dir):
    print('[Model] Training Started')
    print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))

    save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
    ]
    model.fit_generator(data_gen, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks, workers=1)
    print('[Model] Training Completed. Model saved as %s' % save_fname)


def train(news, x, y, epochs, batch_size, save_dir):
    print('[Model] Training Started')
    print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
    save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
    ]

    model.fit([news, x], y, epochs=epochs, batch_size=batch_size, callbacks=callbacks, validation_split=0.07)
    print('[Model] Training Completed. Model saved as %s' % save_fname)


def predict_sequences_multiple(data, window_size, prediction_len):
    # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    print('[Model] Predicting Sequences Multiple...')
    prediction_seqs = []
    for i in range(int(len(data) / prediction_len)):
        curr_frame = data[i * prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[np.newaxis, :, :])[0, 0])
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


if __name__ == '__main__':
    batch_size = 16
    epochs = 64
    models_pool = 'non_attentional_models'

    if not os.path.exists(models_pool):
        os.makedirs(models_pool)

    data = DataLoader(symbols, split=.7, cols=csv_columns, overlap=window)

    # steps_per_epoch = (data.len_train - window) // batch_size

    # train_generator(
    #     data_gen=data.generate_train_batch(
    #         seq_len=window,
    #         batch_size=batch_size,
    #         normalise=True
    #     ), epochs=epochs, batch_size=batch_size, steps_per_epoch=steps_per_epoch, save_dir=models_pool
    # )

    news, x, y = data.get_train_data(seq_len=window, normalise=True)
    news = tf.random.uniform((334, news_messages, news_vector_size, word_vector_size), dtype=tf.float32)
    train(news, x, y, epochs=epochs, batch_size=batch_size, save_dir=models_pool)

    x_test, y_test = data.get_test_data(seq_len=window, normalise=True)
    predictions = predict_sequences_multiple(x_test, window, window)
    plot_results_multiple(predictions, y_test, window)
