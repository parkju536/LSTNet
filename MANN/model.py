import tensorflow as tf
from tensorflow.contrib import layers
import os


class Model(object):
    def __init__(self, config):
        self.config = config
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.regularizer = layers.l2_regularizer(self.config.l2_lambda)
        self.sess = None
        self.saver = None
        self._build_model()

    def _build_model(self):
        self.add_placeholder()
        # make memories
        with tf.variable_scope("memories"):
            memories = tf.concat(tf.split(self.memories, self.config.msteps, axis=1))
            conv_memories = self.conv1d(memories, self.config.kernel_sizes,
                                        self.config.num_filters,
                                        scope="long_term")
            gru_memories = self.gru(conv_memories, scope="long_gru")
            context_memories = self.temporal_attention(gru_memories)
            linear_memories = tf.concat([context_memories, gru_memories[:, -1, :]], axis=1)
            # [b, m, d]
            linear_memories = tf.reshape(linear_memories, [[-1, self.config.msteps, self.config.num_filters * 2]])

        # short term memory
        with tf.variable_scope("short_term"):
            conv = self.conv1d(self.input_x, self.config.kernel_sizes,
                               self.config.num_filters,
                               scope="short_term")
            gru_outputs = self.gru(conv, scope="short_gru")  # [b, t, d]
            context = self.temporal_attention(gru_outputs)  # [b, d]
            last_hidden_states = gru_outputs[:, -1, :]  # [b, d]
            linear_inputs = tf.concat([context, last_hidden_states], axis=1)
        weighted_values = self.get_memory_values(linear_inputs, linear_memories)
        linear_inputs = tf.concat([linear_inputs, weighted_values], axis=1)
        # prediction and loss
        predictions = tf.layers.dense(linear_inputs, self.config.nfeatures,
                                      activation=tf.nn.tanh, use_bias=True,
                                      kernel_regularizer=self.regularizer,
                                      kernel_initializer=layers.xavier_initializer())
        # get auto-regression and add it to prediction from NN
        ar, l2_loss = self.auto_regressive(self.input_x, self.config.ar_lambda)
        self.predictions = predictions + ar
        self.loss = tf.losses.mean_squared_error(labels=self.targets, predictions=self.predictions)
        self.loss += l2_loss
        error = tf.reduce_sum((self.targets - self.predictions) ** 2) ** 0.5
        denom = tf.reduce_sum((self.targets - tf.reduce_mean(self.targets)) ** 2) ** 0.5
        self.rse = error / denom
        if self.config.l2_lambda > 0:
            reg_vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_term = layers.apply_regularization(self.regularizer, reg_vars)
            self.loss += reg_term
        self.add_train_op()
        self.initialize_session()

    def add_placeholder(self):
        self.input_x = tf.placeholder(shape=[None, self.config.nsteps, self.config.nfeatures], dtype=tf.float32,
                                      name="x")
        self.memories = tf.placeholder(shape=[None, self.config.nsteps * self.config.msteps,
                                              self.config.nfeatures], dtype=tf.float32,
                                       name="memories")
        self.targets = tf.placeholder(shape=[None, self.config.nfeatures], dtype=tf.float32, name="targets")
        self.dropout = tf.placeholder(dtype=tf.float32, name="dropout")

    def conv1d(self, inputs, kernel_sizes, num_filters, scope, reuse=False):
        # inputs : [b, t, d]
        with tf.variable_scope(scope, reuse=reuse):
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

    def gru(self, inputs, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            cell = tf.nn.rnn_cell.GRUCell(self.config.num_filters, activation=tf.nn.relu)
            outputs, states = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
            outputs = tf.nn.dropout(states, self.dropout)
        return outputs

    def temporal_attention(self, inputs):
        # use MLP to compute attention score
        # given h_t, attend h_1, h_2,.. h_t-1
        # get last hidden states of GRU
        query = tf.transpose(inputs, [1, 0, 2])[-1]  # [b,t,d] -> [b,d]
        query = tf.expand_dims(query, axis=1)  # [b, d] -> [b, 1, d]
        key = inputs  # [b,t,d]
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
        # [b, t, 1]
        sim_matrix = tf.layers.dense(projection, 1,
                                     activation=None)
        sim_matrix = tf.nn.softmax(sim_matrix, 1)
        # [b, 1, t] dot [b, t, d] -> [b, 1, d]
        context = tf.matmul(tf.transpose(sim_matrix, [0, 2, 1]), inputs)
        context = tf.squeeze(context, axis=-1)
        return context

    def auto_regressive(self, inputs, ar_lambda):
        # y_t,d = sum_i (w_i * y_i,d) + b_d
        w = tf.get_variable(shape=[self.config.nsteps, self.config.nfeatures],
                            initializer=layers.xavier_initializer(),
                            name="w")
        bias = tf.get_variable(shape=[self.config.nfeatures],
                               initializer=tf.zeros_initializer(),
                               name="bias")
        w_ = tf.expand_dims(w, axis=0)
        weighted = tf.reduce_sum(inputs * w_, axis=1) + bias
        l2_loss = ar_lambda * tf.reduce_sum(tf.square(w))
        return weighted, l2_loss

    def get_memory_values(self, query, memories):
        # query : [b, d]
        # memories : [b, m, d]
        query = tf.expand_dims(query, axis=1)
        # dot product [b, 1, d] dot [b, m, d] -> [b,1,m]
        weight = tf.matmul(query, tf.transpose(memories, [0, 2, 1]))
        weight = tf.nn.softmax(weight)
        weighted_values = tf.matmul(weight, memories)

        return weighted_values

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

    def save_session(self, model_name):
        """Saves session = weights"""
        self.saver.save(self.sess, model_name)

    def restore_session(self, dir_model):
        """Reload weights into session
        Args:
            sess: tf.Session()
            dir_model: dir with weights
        """
        self.saver.restore(self.sess, tf.train.latest_checkpoint(dir_model))

    def train(self, batch_x, targets):
        # split numpy array input_x to input_x, memories
        input_x = batch_x[self.config.msteps:]
        memories = batch_x[:self.config.msteps]
        feed_dict = {
            self.input_x: input_x,
            self.memories: memories,
            self.targets: targets,
            self.dropout: self.config.dropout
        }
        output_feed = [self.train_op, self.loss, self.rse, self.mape, self.mae, self.global_step]
        _, loss, res, mape, mae, step = self.sess.run(output_feed, feed_dict)
        return loss, res, mape, mae, step

    def eval(self, batch_x, targets):
        # split numpy array input_x to input_x, memories
        input_x = batch_x[self.config.msteps:]
        memories = batch_x[:self.config.msteps]
        feed_dict = {
            self.input_x: input_x,
            self.memories: memories,
            self.targets: targets,
            self.dropout: 1.0
        }
        output_feed = [self.predictions, self.loss, self.rse, self.mape, self.mae]
        pred, loss, res, mape, mae = self.sess.run(output_feed, feed_dict)
        return pred, loss, res, mape, mae
