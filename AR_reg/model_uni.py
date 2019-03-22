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
        hidden_layers = [16,32]
        self.add_placeholder()
        # get auto-regression 
        self.input = tf.reshape(self.input_x, [-1, self.config.nsteps * self.config.nfeatures])
        #pred = self.mlp_net(self.input, hidden_layers, 1, name="mlp_net")
        self.input = tf.layers.dense(self.input, 32, None, use_bias = True)
        #self.input = tf.layers.dense(self.input, 32, None, use_bias = True)
        
        results = tf.squeeze(tf.layers.dense(self.input, 1, None, use_bias=False))
        #ar, ar_loss = self.auto_regressive(self.input_x, self.config.ar_lambda)
        self.predictions = results
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

    def fully_connect(self, x_tensor, num_outputs):
        """
        Apply a fully connection layer
        """
        # shape_list = x_tensor.get_shape().as_list()
        with tf.name_scope("fully"):
            result = tf.layers.dense(inputs = x_tensor,
                                    units = num_outputs,
                                    activation = tf.nn.relu,
                                    # activation = tf.nn.elu,
                                    kernel_initializer = tf.truncated_normal_initializer())
        return result

    def flatten(self, x_tensor, name):
        """
        Flatten input layer
        """
        # without the 1st param, which is Batch Size
        shape = x_tensor.get_shape().as_list()
        flatten_dim = np.prod(shape[1:])
        with tf.name_scope(name):
            result = tf.reshape(x_tensor, [-1, flatten_dim], name=name)
        return result
    
    def output(self, x_tensor, num_outputs, name):
        """
        Apply a output layer with linear output activation function
        """
        # shape_list = x_tensor.get_shape().as_list()
        # linear output activation function: activation = None
        with tf.name_scope(name):
            result = tf.layers.dense(inputs = x_tensor,
                                    units = num_outputs,
                                    activation = None,
                                    kernel_initializer = tf.truncated_normal_initializer(),
                                    name=name)
        return result


    def mlp_net(self, x, hidden_layers, n_output, name):
        """
        Multilayer Perceptron: 
        multiple fully connect layers
        """
        flatten_layer = hidden_layers
        #flatten_layer = flatten(x, name="flatten_layer")
        with tf.name_scope(name):
            if len(hidden_layers) == 1:
                fully_layer = self.fully_connect(flatten_layer, 16)
            else:
                fully_layer = self.fully_connect(flatten_layer, 16)
                    
                for layer in range(1, len(hidden_layers)):
                    fully_layer = self.fully_connect(fully_layer, 32)
        with tf.name_scope("prediction"):
            pred = self.output(fully_layer, n_output, name="output_layer")

        return pred

    

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
