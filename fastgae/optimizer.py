import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

class OptimizerAE(object):
    """ Optimizer for non-variational autoencoders """
    def __init__(self, preds, labels, labels_louvain, num_nodes, pos_weight, norm, pos_weight_louvain, norm_louvain, clusters):
        preds_sub = preds
        labels_sub = labels
        self.cost_adj =  tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits = preds_sub,
                                                     labels = labels_sub,
                                                     pos_weight = pos_weight))

        # New Louvain term
        #self.cost_louvain =  tf.reduce_mean(
        #    tf.nn.weighted_cross_entropy_with_logits(logits = preds_sub,
        #                                             labels = labels_louvain,
        #                                             pos_weight = pos_weight_louvain))
        #
        self.cost_louvain = tf.reduce_mean(tf.multiply(clusters, labels_louvain))
        self.cost_louvain_neg = tf.reduce_mean(tf.multiply(clusters, (1.0 - labels_louvain)))

        self.cost = self.cost_adj + FLAGS.gamma*self.cost_louvain - FLAGS.gamma2*self.cost_louvain_neg

        # Adam Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate = FLAGS.learning_rate)
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
        self.correct_prediction = \
            tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                     tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))



class OptimizerVAE(object):
    """ Optimizer for variational autoencoders """
    def __init__(self, preds, labels, labels_louvain, model, num_nodes, pos_weight, norm, pos_weight_louvain, norm_louvain, clusters):
        preds_sub = preds
        labels_sub = labels
        self.cost_adj = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits = preds_sub,
                                                     labels = labels_sub,
                                                     pos_weight = pos_weight))
        # Adam Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate = FLAGS.learning_rate)
        # Latent loss
        self.log_lik = self.cost_adj
        self.kl = (0.5 / num_nodes) * \
                  tf.reduce_mean(tf.reduce_sum(1 \
                                               + 2 * model.z_log_std \
                                               - tf.square(model.z_mean) \
                                               - tf.square(tf.exp(model.z_log_std)), 1))
        self.cost_adj -= self.kl

        self.cost_louvain = tf.reduce_mean(tf.multiply(clusters, labels_louvain))
        self.cost_louvain_neg = tf.reduce_mean(tf.multiply(clusters, (1.0 - labels_louvain)))

        # nouveau
        neg_weight = FLAGS.gamma*FLAGS.gamma2

        self.cost = self.cost_adj + FLAGS.gamma*self.cost_louvain - neg_weight*self.cost_louvain_neg

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
        self.correct_prediction = \
            tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                              tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))