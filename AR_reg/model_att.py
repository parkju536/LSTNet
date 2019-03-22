import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np


class Model(object):
    def __init__(self, config):
        self.config = config
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.regularizer = layers.l2_regularizer(self.config.l2_lambda)
        self.sess = None
        self.saver = None
        self._build_model()

    def _build_model(self):
        gru_outputs = []

        self.add_placeholder()
        # get auto-regression 
        
        with tf.variable_scope("inputs"):
            self.input = tf.reshape(self.input_x, [-1, self.config.nsteps * self.config.nfeatures])
            #pred = self.mlp_net(self.input, hidden_layers, 1, name="mlp_net")
            self.input = tf.layers.dense(self.input, 24, None, use_bias = True)
            #self.input = tf.layers.dense(self.input, 6, None, use_bias = True)
        
        result_1 = tf.squeeze(tf.layers.dense(self.input, 1, None, use_bias=False))
        #ar, ar_loss = self.auto_regressive(self.input_x, self.config.ar_lambda)
        
        with tf.variable_scope("attention"):
            for i in range(self.config.nfeatures):
                with tf.variable_scope("Feature_{}".format(i), reuse=False):
                    # input_x : [b, t, nf] -> select one feature [b, t]
                    input_x = self.input_x[:,:,i]
                    input_x = tf.expand_dims(input_x, axis=-1)
                    gru_output = self.gru(input_x, scope="feature_gru")
                    print(gru_output)
                    gru_outputs.append(gru_output)

            # [b, nstep, nf] <- collected last hidden states of gru
            f_gru_outputs = tf.stack(gru_outputs, axis=1)
            #f_gru_outputs = tf.reshape(gru_outputs, [-1, self.config.nsteps * self.config.nfeatures, self.config.attention_size])
            print(f_gru_outputs)

            ### Self-attention
            self.enc = self.multihead_attention(f_gru_outputs,f_gru_outputs,
                                                self.config.attention_size,
                                                self.config.num_heads,
                                                (1-self.config.dropout),
                                                True,"multihead_attention",None)
            
            print("2_",self.enc)

            # pred_ = tf.layers.dense(self.enc, 1,
            #                         activation=tf.nn.tanh, use_bias=True,
            #                         kernel_regularizer=self.regularizer,
            #                         kernel_initializer=layers.xavier_initializer())
            
            pred_ = tf.layers.dense(self.enc, 1,
                                    None, use_bias=True)
            print(pred_)
            pred_ = tf.squeeze(pred_, axis=2)
            result_2 = tf.squeeze(tf.layers.dense(pred_, 1, None, use_bias=False))
            print(pred_)
            
        #self.predictions = result_2
        #self.predictions = 0.9 * result_1 + 0.1 * result_2
        
        self.predictions = result_2
        #self.predictions = tf.squeeze(tf.layers.dense(ar, 1))
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
        #self.loss = ar_loss
        self.add_train_op()
        self.initialize_session()

    def add_placeholder(self):
        self.input_x = tf.placeholder(shape=[None, self.config.nsteps, self.config.nfeatures], dtype=tf.float32,
                                      name="x")
        self.targets = tf.placeholder(shape=[None, ], dtype=tf.float32, name="targets")
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

    def gru(self, inputs, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            cell = tf.nn.rnn_cell.GRUCell(self.config.attention_size, activation=tf.nn.relu)
            outputs, states = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
            states = tf.nn.dropout(states, self.config.dropout)
        return states

    def normalize(self, inputs,
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

    def multihead_attention(self, queries,
                            keys,
                            num_units=None,
                            num_heads=8,
                            dropout_rate=0,
                            is_training=True,
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
            outputs = self.normalize(outputs)  # (N, T_q, C)

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

    

    def add_train_op(self):
        opt = tf.train.AdamOptimizer(self.config.lr)
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

    def train(self, input_x, targets):
        feed_dict = {
            self.input_x: input_x,
            self.targets: targets,
            self.dropout: self.config.dropout
        }
        output_feed = [self.train_op, self.loss, self.rmse, self.rse, self.smape, self.mae, self.global_step]
        _, loss, rmse, res, smape, mae, step = self.sess.run(output_feed, feed_dict)
        return loss, rmse, res, smape, mae, step

    def eval(self, input_x, targets):
        feed_dict = {
            self.input_x: input_x,
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
