import tensorflow.compat.v1 as tf
import numpy as np

class PolicyGradient():

    def __init__(self, param):

        self.lr = None
        self.states = None
        self.actions = None
        self.returns = None
        self.adv = None
        self.pg = None
        self.output_action = None
        self.loss_pg = None
        self.opt_pg = None

        self.sess = tf.Session()

        self.param = param

    def build(self, env):
        """
        This method creates the tf placeholders and operations that form the policy network. As well as its optimization
        """
        tf.compat.v1.disable_eager_execution()

        # Learning rate placeholder
        self.lr = tf.placeholder(tf.float32, shape=None, name='learning_rate')

        # Input placeholders
        self.states = tf.placeholder(tf.float32, shape=[None, env.state_space], name='state')
        self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')
        self.returns = tf.placeholder(tf.float32, shape=(None,), name='returns')
        self.adv = tf.placeholder(tf.float32, shape=(None,), name='advantages')

        # Network architecture. Takes as input states, outputs logits
        self.pg = self.dense_nn(self.states, self.param.layer_shapes + [env.action_space], name='PG_network')

        # Pick an action given the output logits from the network
        self.output_action = tf.squeeze(tf.random.categorical(self.pg, 1))

        # Calculate loss, gradients and update parameters
        with tf.variable_scope('pg_optimize'):
            self.loss_pg = tf.reduce_mean(
                tf.stop_gradient(self.adv) * tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.pg, labels=self.actions), name='loss_pg')

            self.opt_pg = tf.train.AdamOptimizer(self.lr).minimize(self.loss_pg, name='adam_optim_pg')

        init_op = tf.initialize_all_variables()

        self.sess.run(init_op)

    def get_action_e(self, state):
        """
        This method gets as input an state and the policy network outputs the corresponding action sampled from the
        logits
        """
        #see logits
        #convert to arrays everything
        logits = self.sess.run(self.pg, feed_dict={self.states: state})
        if np.random.random() > 0.99:
            action = np.random.randint(0, self.param.action_space)
        else:
            action = self.sess.run(self.output_action, feed_dict={self.states: state})
        return action

    def get_action(self, state):
        logits = self.sess.run(self.pg, feed_dict={self.states: state})
        action = self.sess.run(self.output_action, feed_dict={self.states: state})
        return action

    def optimize_pg(self, states, actions, adv, lr):
        """
        This method run the graph to perform the optimization
        """
        self.sess.run(self.opt_pg, feed_dict={self.states: states, self.actions: actions, self.adv: adv, self.lr: lr})


    def dense_nn(self, inputs, layers_sizes, name="mlp", reuse=False, output_fn=None, dropout_keep_prob=None,
                 batch_norm=False, training=True):

        with tf.variable_scope(name, reuse=reuse):
            out = inputs
            for i, size in enumerate(layers_sizes):
                print("Layer:", name + '_l' + str(i), size)
                if i > 0 and dropout_keep_prob is not None and training:
                    # No dropout on the input layer.
                    out = tf.nn.dropout(out, dropout_keep_prob)

                out = tf.layers.dense(
                    out,
                    size,
                    # Add relu activation only for internal layers.
                    activation=tf.nn.relu if i < len(layers_sizes) - 1 else None,
                    kernel_initializer=tf.keras.initializers.he_uniform(),
                    name=name + '_l' + str(i),
                    reuse=reuse
                )

                if batch_norm:
                    out = tf.layers.batch_normalization(out, training=training)

            if output_fn:
                out = output_fn(out)

        return out
