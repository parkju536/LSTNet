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
        # make memories
        gru_outputs = []
        f_logits = []

        for i in range(self.config.nfeatures):
            with tf.variable_scope("Feature_{}".format(i), reuse=False):
                # input_x : [b, t, nf] -> select one feature [b, t]
                input_x = self.input_x[:,:,i]

                gru_output = self.gru(input_x, scope="feature_gru")
                gru_outputs.append(gru_output)

        # [b, nstep, nf] <- collected last hidden states of gru
        f_gru_outputs = tf.stack(gru_outputs, axis=2)

        ### Self-attention
        self.enc = self.multihead_attention(queris=f_gru_outputs,
                                           key=f_gru_outputs,
                                           num_units=self.config.hidden_units,
                                           num_heads=self.config.num_heads,
                                           dropout_rate=self.config.dropout,
                                           is_training=True,
                                           casuality=False)

        ### Feed Forward
        logits = self.feedforward(self.enc, num_units=[4 * self.config.hidden_units, self.config.hidden_units])

        # weight = tf.get_variable(shape=[self.config.nsteps, self.config.nbins], name="weight", initializer=layers.xavier_initializer())
        # # weight = tf.Variable(tf.random_normal((self.config.nsteps, self.config.nbins), stddev=0.01, dtype='float32'))
        # bias = tf.Variable(tf.constant(0.1, shape=[self.config.nbins]))
        # # [b, t] x [t, nbins] -> [b, nbins]
        # logits = tf.nn.tanh(tf.matmul(input_x, weight) + bias)

        # get predictions
        self.predictions = tf.argmax(f_logits, axis=-1, output_type=tf.int32)
        weights = tf.ones(tf.shape(self.targets))
        self.loss = tf.contrib.seq2seq.sequence_loss(logits=f_logits, targets=self.targets, weights=weights)
        # self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.targets, logits=logits)

        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.predictions, self.targets),dtype=tf.float32))
        self.add_train_op()
        self.initialize_session()

    def add_placeholder(self):
        self.input_x = tf.placeholder(shape=[None, self.config.nsteps, self.config.nfeatures], dtype=tf.float32,
                                      name="x")
        self.targets = tf.placeholder(shape=[None, self.config.nfeatures], dtype=tf.int32, name="targets")
        self.dropout = tf.placeholder(dtype=tf.float32, name="dropout")

    def add_train_op(self):
        opt = tf.train.AdamOptimizer(self.config.lr)
        vars = tf.trainable_variables()
        grads = tf.gradients(self.loss, vars)
        clipped_grads, _ = tf.clip_by_global_norm(grads, self.config.clip)
        self.train_op = opt.apply_gradients(zip(grads, vars), global_step=self.global_step)

    def initialize_session(self):
        """Defines self.sess and initialize the variables"""
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

    def gru(self, inputs, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            cell = tf.nn.rnn_cell.GRUCell(self.config.attention_size)
            outputs, states = tf. nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
            # outputs = tf.nn.dropout(states, self.dropuout)
        return outputs

    def normalize(inputs,
                  epsilon=1e-8,
                  scope="ln",
                  reuse=None):
        '''Applies layer normalization.

        Args:
          inputs: A tensor with 2 or more dimensions, where the first dimension has
            `batch_size`.
          epsilon: A floating number. A very small number for preventing ZeroDivision Error.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

        Returns:
          A tensor with the same shape and data dtype as `inputs`.
        '''
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
            outputs = gamma * normalized + beta

        return outputs

    def multihead_attention(queries,
                            keys,
                            num_units=None,
                            num_heads=8,
                            dropout_rate=0,
                            is_training=True,
                            causality=False,
                            scope="multihead_attention",
                            reuse=None):
        '''Applies multihead attention.

        Args:
          queries: A 3d tensor with shape of [N, T_q, C_q].
          keys: A 3d tensor with shape of [N, T_k, C_k].
          num_units: A scalar. Attention size.
          dropout_rate: A floating point number.
          is_training: Boolean. Controller of mechanism for dropout.
          causality: Boolean. If true, units that reference the future are masked.
          num_heads: An int. Number of heads.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

        Returns
          A 3d tensor with shape of (N, T_q, C)
        '''
        with tf.variable_scope(scope, reuse=reuse):
            # Set the fall back option for num_units
            if num_units is None:
                num_units = queries.get_shape().as_list()[-1]

            # Linear projections
            Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
            K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
            V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

            # Multiplication
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            # Key Masking
            # key_masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # (N, T_k)
            # key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
            # key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

            # paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            # outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

            # Causality = Future blinding
            # if causality:
            #     diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            #     tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            #     masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            #     paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            #     outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

            # Activation
            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

            # Query Masking
            # query_masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
            # query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
            # query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
            # outputs *= query_masks  # broadcasting. (N, T_q, C)

            # Dropouts
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

            # Weighted sum
            outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

            # Residual connection
            outputs += queries

            # Normalize
            outputs = normalize(outputs)  # (N, T_q, C)

        return outputs

    def feedforward(inputs,
                    num_units=[2048, 512],
                    scope="multihead_attention",
                    reuse=None):
        '''Point-wise feed forward net.

        Args:
          inputs: A 3d tensor with shape of [N, T, C].
          num_units: A list of two integers.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

        Returns:
          A 3d tensor with the same shape and dtype as inputs
        '''
        with tf.variable_scope(scope, reuse=reuse):
            # Inner layer
            params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                      "activation": tf.nn.relu, "use_bias": True}
            outputs = tf.layers.conv1d(**params)

            # Readout layer
            params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                      "activation": None, "use_bias": True}
            outputs = tf.layers.conv1d(**params)

            # Residual connection
            outputs += inputs

            # Normalize
            outputs = normalize(outputs)

        return outputs

    def train(self, input_x, targets):
        feed_dict = {
            self.input_x: input_x,
            self.targets: targets,
            self.dropout: self.config.dropout
        }
        output_feed = [self.train_op, self.loss, self.acc, self.global_step]
        _, loss, acc, step = self.sess.run(output_feed, feed_dict)
        return loss, acc, step

    def eval(self, input_x, targets):
        feed_dict = {
            self.input_x: input_x,
            self.targets: targets,
            self.dropout: 1.0
        }
        output_feed = [self.predictions, self.loss, self.acc]
        pred, loss, acc = self.sess.run(output_feed, feed_dict)
        return pred, loss, acc


if __name__ == "__main__":
    from config import Config
    config = Config()
    model = Model(config)
    print("done")
