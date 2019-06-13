import tensorflow as tf
import editdistance


class Model:
    def __init__(self, chars, img_size=(128, 32)):
        tf.reset_default_graph()
        self.chars = chars
        self.snap_id = 0

        self.is_train = tf.placeholder(tf.bool, name="is_train")

        self.imgs = tf.placeholder(tf.float32, shape=(None, img_size[0], img_size[1]))

        self.setup_cnn()
        self.setup_rnn()
        self.setup_ctc()

        self.batches_trained = 0
        self.lr = tf.placeholder(tf.float32, shape=[])
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
        self.session, self.saver = self.setup_tf()

    def setup_cnn(self):
        cnn_in_4d = tf.expand_dims(input=self.imgs, axis=3)

        kernel_values = [5, 5, 3, 3, 3]
        feature_values = [1, 32, 64, 128, 128, 256]
        stride_values = pool_values = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2)]

        pool = cnn_in_4d
        for i in range(len(stride_values)):
            kernel = tf.Variable(
                tf.truncated_normal(
                    [kernel_values[i], kernel_values[i], feature_values[i], feature_values[i + 1]], stddev=0.1
                )
            )
            conv = tf.nn.conv2d(pool, kernel, padding="SAME", strides=(1, 1, 1, 1))
            conv_norm = tf.layers.batch_normalization(conv, training=self.is_train)
            relu = tf.nn.relu(conv_norm)
            pool = tf.nn.max_pool(
                relu,
                (1, pool_values[i][0], pool_values[i][1], 1),
                (1, stride_values[i][0], stride_values[i][1], 1),
                "VALID",
            )

        self.cnn_out_4d = pool

    def setup_rnn(self):
        rnn_in_3d = tf.squeeze(self.cnn_out_4d, axis=[2])

        hidden = 256
        cells = [tf.contrib.rnn.LSTMCell(num_units=hidden, state_is_tuple=True) for _ in range(2)]

        stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        ((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=stacked, cell_bw=stacked, inputs=rnn_in_3d, dtype=rnn_in_3d.dtype
        )

        concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)

        kernel = tf.Variable(tf.truncated_normal([1, 1, hidden * 2, len(self.chars) + 1], stddev=0.1))
        self.rnn_out_3d = tf.squeeze(
            tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding="SAME"), axis=[2]
        )

    def setup_ctc(self):
        self.ctc_in_3d_tbc = tf.transpose(self.rnn_out_3d, [1, 0, 2])
        self.texts = tf.SparseTensor(
            tf.placeholder(tf.int64, shape=[None, 2]), tf.placeholder(tf.int32, [None]), tf.placeholder(tf.int64, [2])
        )

        self.sequence_length = tf.placeholder(tf.int32, [None])

        self.loss = tf.reduce_mean(
            tf.nn.ctc_loss(
                labels=self.texts,
                inputs=self.ctc_in_3d_tbc,
                sequence_length=self.sequence_length,
                ctc_merge_repeated=True,
            )
        )

        self.saved_ctc_Input = tf.placeholder(tf.float32, shape=[32, None, len(self.chars) + 1])

        self.loss_per_element = tf.nn.ctc_loss(
            labels=self.texts,
            inputs=self.saved_ctc_Input,
            sequence_length=self.sequence_length,
            ctc_merge_repeated=True,
        )

        self.decoder = tf.nn.ctc_beam_search_decoder(
            inputs=self.ctc_in_3d_tbc, sequence_length=self.sequence_length, beam_width=50, merge_repeated=False
        )

    def setup_tf(self):
        session = tf.Session()
        saver = tf.train.Saver(max_to_keep=1)
        latest = tf.train.latest_checkpoint("model")

        if latest:
            print("Init with stored values from " + latest)
            saver.restore(session, latest)
        else:
            print("Init with new values")
            session.run(tf.global_variables_initializer())

        return session, saver

    def to_sparse(self, texts):

        indices = []
        values = []
        shape = [len(texts), 0]

        for (batch_element, text) in enumerate(texts):

            label_str = [self.chars.index(c) for c in text]

            if len(label_str) > shape[1]:
                shape[1] = len(label_str)

            for (i, label) in enumerate(label_str):
                indices.append([batch_element, i])
                values.append(label)

        return (indices, values, shape)

    def decoder_output_to_text(self, ctc_output, batch_size):

        encoded_labels = [[] for _ in range(batch_size)]
        decoded = ctc_output[0][0]

        for (idx, idx2d) in enumerate(decoded.indices):
            label = decoded.values[idx]
            batch_element = idx2d[0]
            encoded_labels[batch_element].append(label)

        return [str().join([self.chars[c] for c in label_str]) for label_str in encoded_labels]

    def save(self):
        self.snap_id += 1
        self.saver.save(self.session, "model/snapshot", global_step=self.snap_id)

    def fit(self, X_train, y_train, X_valid, y_valid):
        epoch = 0
        best_char_error = float("inf")
        no_improvement = 0
        early_stopping = 5

        step = 50
        while True:
            epoch += 1
            print("Epoch:", epoch)

            for i in range(0, 25000, step):
                batch_x = X_train[i : i + step]
                batch_y = y_train[i : i + step]
                num_batch = len(batch_x)
                sparse = self.to_sparse(batch_y)
                rate = 0.01 if self.batches_trained < 10 else (0.001 if self.batches_trained < 10000 else 0.0001)
                eval_list = [self.optimizer, self.loss]
                feed_dict = {
                    self.imgs: batch_x,
                    self.texts: sparse,
                    self.sequence_length: [32] * num_batch,
                    self.lr: rate,
                    self.is_train: True,
                }
                _, loss = self.session.run(eval_list, feed_dict)
                self.batches_trained += 1
                print("Batch:", i // step, "Loss:", loss)

            char_error = self.validate(X_valid, y_valid)

            if char_error < best_char_error:
                print("Character error rate improved, save model")
                best_char_error = char_error
                no_improvement = 0
                self.save()
            else:
                print("Character error rate not improved")
                no_improvement += 1

            if no_improvement >= early_stopping:
                print("No more improvement since %d epochs. Training stopped." % early_stopping)
                break

    def predict(self, X):
        num_batch = len(X)
        eval_list = [self.decoder]
        feed_dict = {self.imgs: X, self.sequence_length: [32] * num_batch, self.is_train: False}
        eval_result = self.session.run(eval_list, feed_dict)
        decoded = eval_result[0]
        predicted = self.decoder_output_to_text(decoded, num_batch)
        return predicted

    def validate(self, X, y):
        num_char_error = 0
        num_char_total = 0
        num_word_ok = 0
        num_word_total = 0
        step = 50
        for i in range(0, len(X), step):
            batch_x = X[i : i + step]
            batch_y = y[i : i + step]
            recognized = self.predict(batch_x)

            print("Ground truth -> Recognized")
            for j in range(len(recognized)):
                num_word_ok += 1 if batch_y[j] == recognized[j] else 0
                num_word_total += 1
                dist = editdistance.eval(recognized[j], batch_y[j])
                num_char_error += dist
                num_char_total += len(batch_y[j])
                print(
                    "[OK]" if dist == 0 else "[ERR:%d]" % dist, '"' + batch_y[j] + '"', "->", '"' + recognized[j] + '"'
                )

        char_error = num_char_error / num_char_total
        word_accuracy = num_word_ok / num_word_total
        print("Character error rate: %f%%. Word accuracy: %f%%." % (char_error * 100.0, word_accuracy * 100.0))
        return char_error
