import tensorflow as tf


class CNNLayer:
    def __init__(self, input_imgs, is_train):
        cnn_in_4d = tf.expand_dims(input=input_imgs, axis=3)

        # list of parameters for the layers
        kernel_vals = [5, 5, 3, 3, 3]
        feature_vals = [1, 32, 64, 128, 128, 256]
        stride_vals = pool_vals = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2)]
        num_layers = len(stride_vals)

        # create layers
        pool = cnn_in_4d  # input to first CNN layer
        for i in range(num_layers):
            kernel = tf.Variable(
                tf.truncated_normal([kernel_vals[i], kernel_vals[i], feature_vals[i], feature_vals[i + 1]], stddev=0.1))
            conv = tf.nn.conv2d(pool, kernel, padding='SAME', strides=(1, 1, 1, 1))
            conv_norm = tf.layers.batch_normalization(conv, training=is_train)
            relu = tf.nn.relu(conv_norm)
            pool = tf.nn.max_pool(relu, (1, pool_vals[i][0], pool_vals[i][1], 1),
                                  (1, stride_vals[i][0], stride_vals[i][1], 1), 'VALID')

        self.out = pool


class RNNLayer:
    def __init__(self, prev_layer):
        """create RNN layers and return output of these layers"""
        rnn_in_3d = tf.squeeze(prev_layer, axis=[2])

        # basic cells which is used to build RNN
        num_hidden = 256
        cells = [tf.contrib.rnn.LSTMCell(num_units=num_hidden, state_is_tuple=True) for _ in range(2)]  # 2 layers

        # stack basic cells
        stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        # bidirectional RNN
        # BxTxF -> BxTx2H
        ((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnn_in_3d,
                                                        dtype=rnn_in_3d.dtype)

        # BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
        concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)

        # project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
        kernel = tf.Variable(tf.truncated_normal([1, 1, num_hidden * 2, 80], stddev=0.1))
        self.out = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])


class CTCLayer:
    def __init__(self, prev_layer, labels, seq):
        ctc_in_3d_tbc = tf.transpose(prev_layer, [1, 0, 2])

        self.loss = tf.reduce_mean(
            tf.nn.ctc_loss(labels=labels, inputs=ctc_in_3d_tbc, sequence_length=seq,
                           ctc_merge_repeated=True))

        self.decoder = tf.nn.ctc_beam_search_decoder(inputs=ctc_in_3d_tbc, sequence_length=seq,
                                                     beam_width=50, merge_repeated=False)
