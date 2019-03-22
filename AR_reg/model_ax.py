import tensorflow as tf
from tensorflow.contrib import layers


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
        # get auto-regression 
        #ar, ar_loss = self.auto_regressive(self.input_x, self.config.ar_lambda)
        #self.ar_input = tf.layers.dropout(ar, self.config.dropout)
        #self.ar_result = tf.layers.dense(ar, 1, None, use_bias = True)

        # self.dense_input = tf.reshape(self.input_x, [-1, self.config.nsteps * self.config.nfeatures])
        # self.dense_input = tf.layers.dropout(self.dense_input, self.config.dropout)
        # self.dense_result = tf.layers.dense(self.dense_input, self.config.hidden_size, tf.nn.tanh, use_bias = True,
        #                                      kernel_regularizer=self.regularizer,
        #                                      kernel_initializer=layers.xavier_initializer())

        # self.dense_result = tf.layers.dense(self.dense_result, self.config.hidden_size, tf.nn.tanh, use_bias = True,
        #                                     kernel_regularizer=self.regularizer,
        #                                     kernel_initializer=layers.xavier_initializer())
        
        # with tf.variable_scope("cnn"):
        #     conv = self.conv1d(self.input_x, self.config.kernel_sizes,
        #                        self.config.num_filters,
        #                        scope="short_term")

        # with tf.variable_scope("gru"):
        #     self.gru_outputs = self.gru(conv, scope="short_gru")  # [b, t, d]

        # self.input_ = tf.reshape(self.gru_outputs, shape=[-1, self.config.nsteps * self.config.gru_size])
        # # self.input_ = tf.reshape(conv, [-1, self.config.nsteps * len(self.config.kernel_sizes) * self.config.num_filters])
 
        # self.input_ = tf.layers.dropout(self.input_, self.config.dropout)
        # self.input_ = tf.layers.dense(self.input_, self.config.hidden_size, None, use_bias = True,
        #                             kernel_regularizer=self.regularizer,
        #                             kernel_initializer=layers.xavier_initializer())
        # result = tf.squeeze(tf.layers.dense(self.dense_result, 1, None, use_bias=False))
        # result = tf.squeeze(tf.layers.dense(self.gru_outputs[:,-1,:], 1, tf.nn.tanh, use_bias=False))


        # (3) jinuk
        # with tf.variable_scope("cnn"):
        #     conv = self.conv1d(self.input_x, self.config.kernel_sizes,
        #                        self.config.num_filters,
        #                        scope="short_term")

        with tf.variable_scope("gru"):
            gru_outputs = self.gru(self.input_x, scope="short_gru")  # [b, t, d]    
        # last_hidden = gru_outputs[:,-1,:]

        projection = tf.layers.dense(self.input_c, self.config.hidden_size, tf.nn.tanh, kernel_initializer=layers.xavier_initializer(), use_bias=True)
        # projection = tf.layers.dropout(projection, self.config.dropout)

        attention = self.temporal_attention(gru_outputs, projection)
        linear_input = tf.concat([attention, projection], axis=1)
        # [batch, 48]
        # linear_input = tf.concat([last_hidden, projection], axis=1)
        # linear_input = tf.layers.dense(linear_input, 24, tf.nn.relu, kernel_initializer=layers.xavier_initializer(), use_bias=True)
        result = tf.squeeze(tf.layers.dense(linear_input, 1, None, use_bias=False))

        self.predictions = result

        self.loss = tf.losses.mean_squared_error(labels=self.targets, predictions=self.predictions)

        error = tf.reduce_sum((self.targets - self.predictions) ** 2) ** 0.5
        denom = tf.reduce_sum((self.targets - tf.reduce_mean(self.targets)) ** 2) ** 0.5
        self.rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.targets, self.predictions))))
        self.rse = error / denom
        self.mae = tf.reduce_mean(tf.abs(self.targets - self.predictions))
        self.mape = tf.reduce_mean(tf.abs((self.targets - self.predictions) / self.targets))
        self.smape = tf.reduce_mean(2*tf.abs(self.targets - self.predictions)/(tf.abs(self.targets)+tf.abs(self.predictions)))

        '''
        if self.config.l2_lambda > 0:
            reg_vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_term = layers.apply_regularization(self.regularizer, reg_vars)
            self.loss += reg_term
        '''
        #self.loss += ar_loss

        self.add_train_op()
        self.initialize_session()

    def add_placeholder(self):
        self.input_x = tf.placeholder(shape=[None, self.config.nsteps, self.config.nfeatures], dtype=tf.float32, name="x")
        self.input_c = tf.placeholder(shape=[None, self.config.nfeatures-1], dtype=tf.float32, name="c")                              
        self.targets = tf.placeholder(shape=[None,], dtype=tf.float32, name="targets")
        self.dropout = tf.placeholder(dtype=tf.float32, name="dropout")

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
        ar_loss = ar_lambda * tf.reduce_sum(tf.square(w))
        return weighted, ar_loss

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
            cell = tf.nn.rnn_cell.GRUCell(self.config.gru_size, activation=tf.nn.relu)
            outputs, states = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
            # output = tf.nn.dropout(outputs, self.dropout)
        return outputs

    def temporal_attention(self, gru_outputs, projection):
        # use MLP to compute attention score
        # given dense projection, attend h_1, h_2,.. h_t
        # gru_outputs : [batchs, time, gru_ize]
        # projection : [batch, hidden_size] -> [batch, 1, hidden_size]
        projection = tf.tile(tf.expand_dims(projection, axis=1), [1, self.config.nsteps, 1])
        linear_input = tf.concat([gru_outputs, projection], axis=-1)
        
        # sim : [batch, time, attention_size]
        sim = tf.layers.dense(linear_input, self.config.attention_size, activation=tf.nn.tanh, use_bias=True,
                                kernel_initializer=layers.xavier_initializer())
        # [b, t, 1]
        sim_matrix = tf.layers.dense(sim, 1, activation=None)
        sim_matrix = tf.nn.softmax(sim_matrix, 1)
        # [b, 1, t] dot [b, t, d] -> [b, 1, d]
        context = tf.matmul(tf.transpose(sim_matrix, [0, 2, 1]), gru_outputs)
        context = tf.squeeze(context, axis=1)
        
        return context

    def add_train_op(self):
        opt = tf.train.AdamOptimizer(self.config.lr)
        # opt = tf.train.RMSPropOptimizer(self.config.lr)
        vars = tf.trainable_variables()
        grads = tf.gradients(self.loss, vars)
        clipped_grads, _ = tf.clip_by_global_norm(grads, self.config.clip)
        self.train_op = opt.apply_gradients(zip(grads, vars), global_step=self.global_step)

    def initialize_session(self):
        """Defines self.sess and initialize the variables"""
        if not self.config.allow_gpu:
            config = tf.ConfigProto(device_count={'GPU': 0})
        else:
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

    def train(self, input_x, input_c, targets):
        feed_dict = {
            self.input_x: input_x,
            self.input_c: input_c,
            self.targets: targets,
            self.dropout: self.config.dropout
        }
        output_feed = [self.train_op, self.loss, self.rmse, self.rse, self.smape, self.mae, self.global_step]
        _, loss, rmse, res, smape, mae, step = self.sess.run(output_feed, feed_dict)
        return loss, rmse, res, smape, mae, step

    def eval(self, input_x, input_c, targets):
        feed_dict = {
            self.input_x: input_x,
            self.input_c: input_c,
            self.targets: targets,
            self.dropout: 1.0
        }
        output_feed = [self.predictions, self.loss, self.rmse, self.rse, self.smape, self.mae]
        pred, loss, rmse, res, smape, mae = self.sess.run(output_feed, feed_dict)
        return pred, loss, rmse, res, smape, mae

if __name__ == "__main__":
    from config import Config
    config = Config()
    model = Model(config)
    print("done")
