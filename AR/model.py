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
        f_logits = []

        for i in range(self.config.nfeatures):
            with tf.variable_scope("Feature_{}".format(i), reuse=False):
                # input_x : [b, t, nf] -> select one feature [b, t]
                input_x = self.input_x[:,:,i]
                
                weight = tf.get_variable(shape=[self.config.nsteps, self.config.nbins], name="weight", initializer=layers.xavier_initializer())
                # weight = tf.Variable(tf.random_normal((self.config.nsteps, self.config.nbins), stddev=0.01, dtype='float32'))
                bias = tf.Variable(tf.constant(0.1, shape=[self.config.nbins]))
                # [b, t] x [t, nbins] -> [b, nbins]
                logits = tf.nn.tanh(tf.matmul(input_x, weight) + bias)

            f_logits.append(logits)

        f_logits = tf.stack(f_logits, axis=1)
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
