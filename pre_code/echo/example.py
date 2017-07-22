import requests
import numpy as np
import tensorflow as tf

from pre_code.echo.esn_cell import ESNCell


def MackeyGlass(tr_size=1000, washout_size=100, units=1000, connectivity=1, scale=0.3, elements=2000):
    data_str = requests.get(
        'http://minds.jacobs-university.de/sites/default/files/uploads/mantas/code/MackeyGlass_t17.txt').content
    data_str = data_str.splitlines()[:elements]
    data = [float(string) for string in data_str]
    data_t = tf.reshape(tf.constant(data), [1, elements, 1])
    esn = ESNCell(num_units=units, connectivity=connectivity, wr2_scale=scale)

    print("Building graph...")
    outputs, final_state = tf.nn.dynamic_rnn(esn, data_t, dtype=tf.float32)
    washed = tf.squeeze(tf.slice(outputs, [0, washout_size, 0], [-1, -1, -1]))

    with tf.Session() as S:
        S.run(tf.global_variables_initializer())

        print("Computing embeddings...")
        res = S.run(washed)

        print("Computing direct solution...")
        state = np.array(res)
        tr_state = np.mat(state[:tr_size])
        ts_state = np.mat(state[tr_size:])
        wout = np.transpose(
            np.mat(data[washout_size + 1:tr_size + washout_size + 1]) * np.transpose(np.linalg.pinv(tr_state)))

        print("Testing performance...")
        ts_out = np.mat((np.transpose(ts_state * wout).tolist())[0][:-1])
        ts_y = np.mat(data[washout_size + tr_size + 1:])
        print("ts_y", ts_y)
        print("ts_out: ", ts_out)
        ts_mse = np.mean(np.square(ts_y - ts_out))

    print("Test MSE: " + str(ts_mse))


if __name__ == "__main__":
    MackeyGlass()
