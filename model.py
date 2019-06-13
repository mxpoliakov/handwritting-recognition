import time
from random import shuffle

import tensorflow as tf
import editdistance

from layers import CNNLayer, RNNLayer, CTCLayer


class Model:

    img_size = (128, 32)
    max_text_len = 32

    def __init__(self):
        self.batches_trained = 0

        self.setup_placeholders()
                
        cnn = CNNLayer(self.input_imgs, self.is_train)
        rnn = RNNLayer(cnn.out)
        ctc = CTCLayer(rnn.out, self.gt_texts, self.seq_len)
      
        self.loss = ctc.loss
        self.decoder = ctc.decoder

        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        self.initialize_characters_list()    
        
        self.setup_tf()
    
    def setup_placeholders(self):
        self.is_train = tf.placeholder(tf.bool, name='is_train')
        
        self.input_imgs = tf.placeholder(tf.float32, shape=(None, Model.img_size[0], Model.img_size[1]))

        self.gt_texts = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2]), tf.placeholder(tf.int32, [None]),
                                        tf.placeholder(tf.int64, [2]))
        
        self.seq_len = tf.placeholder(tf.int32, [None])
        
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
    
    def initialize_characters_list(self):
        self.char_list = sorted(map(lambda x:chr(x), set(range(32,123)).difference(set([36,37,60,61,62,64]+list(range(91,97))))))

    def setup_tf(self):
        self.snap_id = 0
        
        self.session = tf.Session()  # TF session

        self.saver = tf.train.Saver(max_to_keep=1)  # saver saves model to file
        model_dir = 'model/'
        latest_snapshot = tf.train.latest_checkpoint(model_dir)  # is there a saved model?

        # load saved model if available
        if latest_snapshot:
            print('Init with stored values from ' + latest_snapshot)
            self.saver.restore(self.session, latest_snapshot)
        else:
            print('Init with new values')
            self.session.run(tf.global_variables_initializer())

    def to_sparse(self, texts):
        indices, values, sizes = zip(*[[[i, j], self.char_list.index(texts[i][j]), len(texts[i])] for i in range(len(texts)) for j in range(len(texts[i]))])

        return (indices, values, [len(texts), max(sizes)])

    def decoder_output_to_text(self, ctc_output, batch_size):
        encoded_label_strs = [[] for _ in range(batch_size)]
        
        decoded = ctc_output[0][0]

        list(map(lambda pair: encoded_label_strs[pair[1][0]].append(decoded.values[pair[0]]), enumerate(decoded.indices)))
        return [str().join([self.char_list[c] for c in labelStr]) for labelStr in encoded_label_strs]

    def save(self):
        self.snap_id += 1
        self.saver.save(self.session, 'model/snapshot', global_step=self.snap_id)
    
    def fit(self, X_tr, y_tr, X_valid, y_valid, epoches=False, early=False):
        epoch = 0
        best_char_error_rate = float('inf')
        no_improvement_since = 0
        step = 50

        data = list(zip(X_tr, y_tr))

        while True:
            epoch += 1
            print('Epoch:', epoch)
            epoch_start = time.time()
            shuffle(data)
            X_tr, y_tr = zip(*data)

            for i in range(0, 25000, step):
                batch_x = X_tr[i:i+step]
                batch_y = y_tr[i:i+step]
                num_elements = len(batch_x)
                sparse = self.to_sparse(batch_y)
                rate = 0.01 if self.batches_trained < 10 else (
                  0.001 if self.batches_trained < 10000 else 0.0001)
                eval_list = [self.optimizer, self.loss]
                feed_dict = {self.input_imgs: batch_x,
                             self.gt_texts: sparse,
                             self.seq_len: [Model.max_text_len] * num_elements,
                             self.learning_rate: rate,
                             self.is_train: True
                             }
                self.session.run(eval_list, feed_dict)
                self.batches_trained += 1

            char_error_rate = self.validate(X_valid, y_valid)

            print("Time passed:", time.time() - epoch_start)

            if char_error_rate < best_char_error_rate:
                best_char_error_rate = char_error_rate
                no_improvement_since = 0
                self.save()
            else:
                no_improvement_since += 1

            # stop training if no more improvement in the last x epochs
            if early and no_improvement_since >= early:
                print('No more improvement since %d epochs. Training stopped.' % early)
                break

            if epoches and epoch >= epoches:
                print("Epoches limit riched. Training stopped.")
                break
    
    def predict(self, X):
        num_elements = len(X)
        eval_list = [self.decoder]
        feed_dict = {self.input_imgs: X,
                     self.seq_len: [Model.max_text_len] * num_elements,
                     self.is_train: False
                     }
        eval_res = self.session.run(eval_list, feed_dict)
        decoded = eval_res[0]
        predicted = self.decoder_output_to_text(decoded, num_elements)
        return predicted
    
    def validate(self, X, y):
        recognized = self.predict(X)

        batch_char_errs = [editdistance.eval(recognized[i], y[i]) for i in range(len(X))]

        char_error_rate = sum(batch_char_errs) / sum([len(el) for el in y])
        word_accuracy = batch_char_errs.count(0) / len(y)

        print('Character error rate: %f%%. Word accuracy: %f%%.' % (char_error_rate*100.0, word_accuracy*100.0))
        return char_error_rate
