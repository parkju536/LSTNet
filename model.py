import tensorflow as tf
from tensorflow.contrib import layers
import os
from data_utils import batch_loader


class LSTNet(object):
    def __init__(self, config):
        self.config = config
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.regularizer = layers.l2_regularizer(self.config.l2_lambda)
        self.sess = None
        self.saver = None
        self._build_model()

    def _build_model(self):
        self.add_placeholder()
        conv = self.conv1d(self.input_x, self.config.kernel_sizes, self.config.num_filters)
        gru_outputs = self.gru(conv)
        contexts = self.temporal_attention(gru_outputs)
        linear_inputs = tf.concat([contexts, gru_outputs], axis=2)
        # prediction and loss
        predictions = tf.layers.dense(linear_inputs, self.config.nfeatures,
                                      activation=None, use_bias=True,
                                      kernel_regularizer=self.regularizer,
                                      kernel_initializer=layers.xavier_initializer())
        # get the last prediction
        self.prediction = predictions[:, -1, :]
        self.loss = tf.losses.mean_squared_error(labels=self.targets, predictions=self.prediction)
        error = tf.reduce_sum((self.targets - self.prediction) ** 2) ** 0.5
        denom = tf.reduce_sum((self.targets - tf.reduce_mean(self.targets)) ** 2) ** 0.5
        self.rse = error / denom
        if self.config.l2_lambda > 0:
            reg_vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_term = layers.apply_regularization(self.regularizer, reg_vars)
            self.loss += reg_term
        self.add_train_op()
        self.initialize_session()

    def add_placeholder(self):
        self.input_x = tf.placeholder(shape=[None, None, self.config.nfeatures], dtype=tf.float32, name="x")
        self.targets = tf.placeholder(shape=[None, self.config.nfeatures], dtype=tf.float32, name="targets")
        self.dropout = tf.placeholder(dtype=tf.float32, name="dropout")

    def conv1d(self, inputs, kernel_sizes, num_filters):
        # inputs : [b, t, d]
        conv_lst = []
        for i in range(len(kernel_sizes)):
            kernel_size = kernel_sizes[i]
            conv = tf.layers.conv1d(inputs, num_filters, kernel_size,
                                    use_bias=True,
                                    kernel_regularizer=self.regularizer,
                                    name="filter_{}".format(i),
                                    padding="same",
                                    kernel_initializer=layers.variance_scaling_initializer(), )
            conv = tf.nn.relu(conv)
            conv_lst.append(conv)
        # outputs : [b, t, num_filters * len(kernel_sizes)]
        outputs = tf.concat(conv_lst, axis=2)
        outputs = tf.nn.dropout(outputs, self.dropout)
        return outputs

    def gru(self, inputs):
        cell = tf.nn.rnn_cell.GRUCell(inputs, activation=tf.nn.relu)
        outputs, states = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
        outputs = tf.nn.dropout(outputs, self.dropout)
        return outputs

    def temporal_attention(self, inputs):
        # use MLP to compute attention score
        # given h_t, attend h_1, h_2,.. h_t-1
        query = tf.expand_dims(inputs, axis=2)  # [b,t,d] -> [b,t,1,d]
        key = tf.expand_dims(inputs, axis=1)  # [b,t,d] -> [b,1,t,d]
        query = tf.layers.dense(query, self.config.attention_size,
                                activation=None, use_bias=False,
                                kernel_initializer=layers.xavier_initializer(),
                                kernel_regularizer=self.regularizer)

        key = tf.layers.dense(key, self.config.attention_size,
                              activation=None, use_bias=False,
                              kernel_initializer=layers.xavier_initializer(),
                              kernel_regularizer=self.regularizer)

        bias = tf.get_variable(shape=[self.config.attention_size],
                               initializer=tf.zeros_initializer(),
                               name="attention_bias")
        projection = tf.nn.tanh(query + key + bias)
        # [b, t, t, 1] -> [b, t, t]
        sim_matrix = tf.layers.dense(projection, 1,
                                     activation=None)
        sim_matrix = tf.squeeze(sim_matrix, axis=-1)
        # mask for future blinding
        ones = tf.ones_like(sim_matrix)
        # get lower triangular matrix
        mask = tf.matrix_band_part(ones, -1, 0)
        paddings = tf.ones_like(mask) * (-2 ** 32)
        sim_matrix = tf.where(tf.equal(mask, 0), paddings, sim_matrix)
        sim_matrix = tf.nn.softmax(sim_matrix)
        contexts = tf.matmul(sim_matrix, inputs)
        return contexts

    def add_train_op(self):
        opt = tf.train.AdamOptimizer(self.config.lr)
        vars = tf.trainable_variables()
        grads = tf.gradients(self.loss, vars)
        clipped_grads, _ = tf.clip_by_global_norm(grads, self.config.clip)
        self.train_op = opt.apply_gradients(zip(grads, vars), global_step=self.global_step)

    def initialize_session(self):
        """Defines self.sess and initialize the variables"""
        print("Initializing tf session")
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def save_session(self, path):
        """Saves session = weights"""
        if not os.path.exists(path):
            os.makedirs(path)
        self.saver.save(self.sess, path)

    def restore_session(self, dir_model):
        """Reload weights into session
        Args:
            sess: tf.Session()
            dir_model: dir with weights
        """
        print("Reloading the latest trained model...")
        self.saver.restore(self.sess, dir_model)

    def train(self, train_batches):
        batches = batch_loader(train_batches, self.config.batch_size)
        for i in range(self.config.num_epochs):
            epoch = i + 1
            for batch in batches:
                input_x, targets = zip(*batch)
                feed_dict = {
                    self.input_x: input_x,
                    self.targets: targets,
                    self.dropout: self.config.dropout
                }
                output_feed = [self.train_op, self.loss, self.rse, self.global_step]
                _, loss, rse, step = self.sess.run(output_feed, feed_dict)
                if step % 100 == 0:
                    print("epoch :%d, step: %d, rse: %.4f loss : %.4f" % (epoch, step, rse, loss))
