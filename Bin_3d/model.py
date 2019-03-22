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
        f_inputs = []

        for i in range(self.config.nfeatures):
            with tf.variable_scope("memories_{}".format(i), reuse=False):
                # self.memories : [b, (n+1)*m, nf]
                memories = tf.concat(tf.split(self.memories, self.config.msteps, axis=1), axis=0) # [b*m, n+1, nf]
                #conv_memories : [b*m, n+1, num_filter * len(kernel_sizes)]
                conv_memories = self.conv1d(memories, self.config.kernel_sizes, 
                                            self.config.num_filters,
                                            scope="long_term_{}".format(i))
                # gru : [b*m, n+1, attention_size]
                gru_memories = self.gru(conv_memories, scope="long_gru_{}".format(i))
                
                context_memories = self.temporal_attention(gru_memories)
                # [b*m, 2d]
                linear_memories = tf.concat([context_memories, gru_memories[:,-1,:]],axis=1)
                # [b,m,2d]
                linear_memories = tf.reshape(linear_memories, [-1, self.config.msteps, self.config.attention_size * 2])
                
                
            # short term memory
            with tf.variable_scope("short_term_{}".format(i), reuse=False):
                conv = self.conv1d(self.input_x, self.config.kernel_sizes,
                                    self.config.num_filters,
                                    scope="short_term_{}".format(i))
                gru_outputs = self.gru(conv, scope="short_gru_{}".format(i))  # [b,t,d]
                context = self.temporal_attention(gru_outputs)  # [b,d]
                last_hidden_states = gru_outputs[:,-1,:]        # [b,d]
                # ar_1 = self.ar_1(self.input_x)
                linear_inputs = tf.concat([last_hidden_states, context], axis=-1) # [b,2d]

            weighted_values = self.get_memory_values(linear_inputs, linear_memories)
            linear_inputs = tf.concat([linear_inputs, weighted_values], axis=1)   # [b,3d]

            # ar_1 concat
            ar = self.ar_1(self.input_x)
            # [b, d]
            linear_inputs = tf.concat([linear_inputs, ar], axis=1)
            # [b, nbins]
            linear_inputs = tf.layers.dense(linear_inputs, self.config.nbins, 
                                    activation=None, use_bias=True,
                                    kernel_initializer=layers.xavier_initializer(),
                                    kernel_regularizer=self.regularizer)

            f_inputs.append(linear_inputs)

        # outputs : [b,nf,nbins]   
        f_inputs = tf.stack(f_inputs, axis=1)
        # f_inputs = tf.nn.dropout(f_inputs, self.dropout)
        
        # multihead_logits [b, nf, nbins]
        #multihead_logits = self.multihead_outputs(linear_inputs)  

        # get predictions
        self.predictions = tf.argmax(f_inputs, axis=-1, output_type=tf.int32)

        weights = tf.ones(tf.shape(self.targets))
        self.loss = tf.contrib.seq2seq.sequence_loss(logits=f_inputs, targets=self.targets, weights=weights)
        # self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.targets, logits=logits)

        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.predictions, self.targets),dtype=tf.float32))


    def add_placeholder(self):
        self.input_x = tf.placeholder(shape=[None, self.config.nsteps, self.config.nfeatures], dtype=tf.float32,
                                      name="x")
        self.targets = tf.placeholder(shape=[None, self.config.nfeatures], dtype=tf.int32, name="targets")
        self.memories = tf.placeholder(shape=[None, (self.config.nsteps+1) * self.config.msteps, self.config.nfeatures], dtype=tf.float32, name="memories")
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


    def get_memory_values(self, query, memories):
        # query : [b,d]
        # memories : [b,m,d]
        query = tf.expand_dims(query, axis=1)
        # dot product [b,1,d] dot [b,m,d] -> [b,1,m]
        weight = tf.matmul(query, tf.transpose(memories, [0,2,1]))
        weight = tf.nn.softmax(weight)
        # weighted_values : [b,1,d] -> [b,d]
        weighted_values = tf.matmul(weight, memories)
        weighted_values = tf.squeeze(weighted_values, axis=1)

        return weighted_values

    def multihead_outputs(self, inputs):
        # inputs : [b,d]
        logits_lst = []
        for i in range(self.config.nfeatures):
            # logits [b,nbins]
            logits = tf.layers.dense(inputs, self.config.nbins,
                                    activation=None, use_bias=True,
                                    name="logits_{}".format(i),
                                    kernel_initializer=layers.xavier_initializer(),
                                    kernel_regularizer=self.regularizer)
            logits_lst.append(logits)
        # outputs : [b,nf,nbins]
        outputs = tf.stack(logits_lst, axis=1)
        outputs = tf.nn.dropout(outputs, self.dropout)

        return outputs

    def gru(self, inputs, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            cell = tf.nn.rnn_cell.GRUCell(self.config.attention_size, activation=tf.nn.relu)
            outputs, states = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
            outputs = tf.nn.dropout(outputs, self.dropout)
        return outputs

    def ar_1(self, inputs):
        # inputs [b,t,d] -> [b,d]
        inputs = inputs[:,-1,:]
        outputs = tf.layers.dense(inputs, self.config.attention_size, 
                                activation=tf.nn.relu, use_bias=True,
                                kernel_initializer=layers.xavier_initializer(),
                                kernel_regularizer=self.regularizer)
        outputs = tf.nn.dropout(outputs, self.dropout)
        # outputs = [b,d]
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
        context = tf.squeeze(context, axis=1)
        return context

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
