import numpy as np
import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops


class Reward(object):
    def __init__(self, lstm, update_rate):
        self.lstm = lstm
        self.update_rate = update_rate
        self.num_emb = self.lstm.num_vocabulary
        self.batch_size = self.lstm.batch_size
        self.emb_dim = self.lstm.emb_dim
        self.hidden_dim = self.lstm.hidden_dim
        self.sequence_length = self.lstm.sequence_length
        self.start_token = tf.identity(self.lstm.start_token)
        self.learning_rate = self.lstm.learning_rate

        self.g_embeddings = tf.identity(self.lstm.g_embeddings)
        self.g_recurrent_unit = self.create_recurrent_unit()  # maps h_tm1 to h_t for generator
        self.g_output_unit = self.create_output_unit()  # maps h_t to o_t (output token logits)



    def get_reward(self, sess, input_x, rollout_num, discriminator, dis_data_loader):
        rewards = []
        seq_len = len(input_x[0])
        for i in range(rollout_num):

            for given_num in range(1, seq_len):
                feed = {self.x: input_x, self.given_num: given_num}
                samples = sess.run(self.gen_x, feed)
                feed = {discriminator.input_x: samples, discriminator.input_ref: dis_data_loader.get_reference()}
                scores = sess.run(discriminator.ypred_for_auc, feed)
                ypred = np.array([item for item in scores])
                if i == 0:
                    rewards.append(ypred)
                else:
                    rewards[given_num - 1] += ypred

            # the last token reward
            feed = {discriminator.input_x: input_x, discriminator.input_ref: dis_data_loader.get_reference()}
            scores = sess.run(discriminator.ypred_for_auc, feed)
            ypred = np.array([item for item in scores])
            if i == 0:
                rewards.append(ypred)
            else:
                rewards[seq_len - 1] += ypred

        rewards = np.transpose(np.array(rewards)) / (1.0 * rollout_num)  # batch_size x seq_length
        return rewards


