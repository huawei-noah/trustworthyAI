import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell, MultiRNNCell, DropoutWrapper
from tensorflow.contrib import distributions as distr

from ..encoder import TransformerEncoder
# from config_graph import get_config, print_config
# from dataset_graph import DataGenerator


def multihead_attention(inputs, num_units=None, num_heads=16, dropout_rate=0.1, is_training=True):
 
    with tf.variable_scope("multihead_attention", reuse=None):
        
        # Linear projections
        Q = tf.layers.dense(inputs, num_units, activation=tf.nn.relu) # [batch_size, seq_length, n_hidden]
        K = tf.layers.dense(inputs, num_units, activation=tf.nn.relu) # [batch_size, seq_length, n_hidden]
        V = tf.layers.dense(inputs, num_units, activation=tf.nn.relu) # [batch_size, seq_length, n_hidden]
        
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # [batch_size, seq_length, n_hidden/num_heads]
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # [batch_size, seq_length, n_hidden/num_heads]
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # [batch_size, seq_length, n_hidden/num_heads]
 
        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # num_heads*[batch_size, seq_length, seq_length]
        
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
  
        # Activation
        outputs = tf.nn.softmax(outputs) # num_heads*[batch_size, seq_length, seq_length]
          
        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
               
        # Weighted sum
        outputs = tf.matmul(outputs, V_) # num_heads*[batch_size, seq_length, n_hidden/num_heads]
        
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # [batch_size, seq_length, n_hidden]
              
        # Residual connection
        outputs += inputs # [batch_size, seq_length, n_hidden]
              
        # Normalize
        outputs = tf.layers.batch_normalization(outputs, axis=2, training=is_training, name='ln', reuse=None)  # [batch_size, seq_length, n_hidden]
 
    return outputs
 
 
# Apply point-wise feed forward net to a 3d tensor with shape [batch_size, seq_length, n_hidden]
# Returns: a 3d tensor with the same shape and dtype as inputs
 
def feedforward(inputs, num_units=[2048, 512], is_training=True):
 
    with tf.variable_scope("ffn", reuse=None):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1, "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1, "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        
        # Residual connection
        outputs += inputs
        
        # Normalize
        outputs = tf.layers.batch_normalization(outputs, axis=2, training=is_training, name='ln', reuse=None)  # [batch_size, seq_length, n_hidden]
    
    return outputs
 
 
 
class TransformerDecoder(object):
 
    def __init__(self, config, is_train):
        self.batch_size = config.batch_size # batch size
        self.max_length = config.max_length # input sequence length (number of cities)
        self.input_dimension = config.hidden_dim#input_dimension*2+1 # dimension of input, multiply 2 for expanding dimension to input complex value to tf, add 1 high priority token, 1 pointing
 
        self.input_embed = config.hidden_dim # dimension of embedding space (actor)
        self.num_heads = config.num_heads
        self.num_stacks = config.num_stacks
        self.max_length = config.max_length
 
        self.initializer = tf.contrib.layers.xavier_initializer() # variables initializer
 

        self.is_training = is_train




        self.samples = []



        ########################################
        ########## Initialize process ##########
        ########################################




        # Keep track of visited cities
        self.mask = 0
        self.mask_scores = []

        self.entropy = []

 
 
    def decode(self, inputs):
 
        # Tensor blocks holding the input sequences [Batch Size, Sequence Length, Features]
        # self.input_ = tf.placeholder(tf.float32, [self.batch_size, self.max_length, self.input_dimension], name="input_raw")
 
        # with tf.variable_scope("embedding_MCS"):
        #   # Embed input sequence
        #   W_embed =tf.get_variable("weights",[1,self.input_dimension, self.input_embed], initializer=self.initializer)
        #   self.embedded_input = tf.nn.conv1d(inputs, W_embed, 1, "VALID", name="embedded_input")
        #   # Batch Normalization
        #   self.enc = tf.layers.batch_normalization(self.embedded_input, axis=2, training=self.is_training, name='layer_norm', reuse=None)



        all_user_embedding = tf.reduce_mean(inputs, 1)
        inputs_with_all_user_embedding = tf.concat([inputs,
                                        tf.tile(tf.expand_dims(all_user_embedding,1), [1, self.max_length ,1])], -1)
 
        with tf.variable_scope("embedding_MCS"):
          # Embed input sequence
          W_embed =tf.get_variable("weights",[1,self.input_embed , self.input_embed], initializer=self.initializer) #self.input_dimension*2
          self.embedded_input = tf.nn.conv1d(inputs, W_embed, 1, "VALID", name="embedded_input")
          # Batch Normalization
          self.enc = tf.layers.batch_normalization(self.embedded_input, axis=2, training=self.is_training, name='layer_norm', reuse=None)
        
        with tf.variable_scope("stack_MCS"):
          # Blocks
          for i in range(self.num_stacks): # num blocks
              with tf.variable_scope("block_{}".format(i)):
                  ### Multihead Attention
                  self.enc = multihead_attention(self.enc, num_units=self.input_embed, num_heads=self.num_heads, dropout_rate=0.0, is_training=self.is_training)

                  ### Feed Forward
                  self.enc = feedforward(self.enc, num_units=[self.input_embed, self.input_embed], is_training=self.is_training)
 
          # Return the output activations [Batch size, Sequence Length, Num_neurons] as tensors.
          # self.encoder_output = self.enc ### NOTE: encoder_output is the ref for attention ###
          # Readout layer
          params = {"inputs": self.enc, "filters": self.max_length, "kernel_size": 1, "activation": None, "use_bias": True}


          self.adj_prob = tf.layers.conv1d(**params)


          for i in range(self.max_length):
              # Multinomial distribution
              # prob_test = tf.convert_to_tensor(np.array([[0,0.9,0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] for i in range(32)]), dtype = tf.float32)
              # prob_test = tf.Print(prob_test, ['prob_test  value is', prob_test], summarize=100)

            position = tf.ones([inputs.shape[0]]) * i
            position = tf.cast(position, tf.int32)

            # Update mask
            self.mask = tf.one_hot(position, self.max_length)


            masked_score = self.adj_prob[:,i,:] - 100000000.*self.mask
            prob = distr.Bernoulli(masked_score)#probs input probability, logit input log_probability

            sampled_arr = prob.sample() # Batch_size, seqlenght for just one node

            self.samples.append(sampled_arr)
            self.mask_scores.append(masked_score)

            self.entropy.append(prob.entropy())



          # self.mcs_prob = tf.Print(self.mcs_prob, ['self.mcs_prob  value is', self.mcs_prob], summarize=100)

          # self.mcs_sampling=tf.cast(tf.arg_max(self.mcs_prob, -1), tf.int32)

          return self.samples, self.mask_scores, self.entropy
 



if __name__ == '__main__':
    config, _ = get_config()
    print('check:', config.batch_size, config.max_length, config.input_dimension)
    input_ = tf.placeholder(tf.float32, [config.batch_size, config.max_length, config.input_dimension],
                                 name="input_channel")

    Encoder = TransformerEncoder(config, True)
    encoder_output = Encoder.encode(input_)

    # Ptr-net returns permutations (self.positions), with their log-probability for backprop
    ptr = Pointer_decoder(encoder_output, config)
    samples, logits_for_rewards = ptr.loop_decode_for_test()

    graphs_gen = tf.stack(samples)
    graphs_gen = tf.transpose(graphs_gen, [1, 0, 2])
    graphs_gen = tf.cast(graphs_gen, tf.float32)
    logits_for_rewards = tf.stack(logits_for_rewards)
    logits_for_rewards = tf.transpose(logits_for_rewards, [1, 0, 2])
    log_probss = tf.nn.sigmoid_cross_entropy_with_logits(labels=graphs_gen, logits=logits_for_rewards)
    reward_probs = tf.reduce_sum(log_probss, axis=[1,2])

    rewards = tf.reduce_sum(tf.abs(graphs_gen), axis=[1, 2])

    # s, i = self.decoder_initial_state, tf.cast(self.decoder_first_input, tf.float32)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        solver = []
        training_set = DataGenerator(solver, True)

        nb_epoch = 2

        for i in range(nb_epoch):
            input_batch = training_set.train_batch(config.batch_size, config.max_length, config.input_dimension)
            # self.decoder_initial_state, tf.cast(self.decoder_first_input, tf.float32)
            # ss, ll = sess.run([samplptres,log_softmax], feed_dict={input_: input_batch})
            #ss, ll, tt = sess.run([tf.shape(ptr.decoder_initial_state), tf.shape(ptr.decoder_first_input), tf.shape(encoder_output)],
            #                 feed_dict={input_: input_batch})

            # print(sess.run([tf.shape(ptr.s_check0), tf.shape(ptr.i_check0), tf.shape(ptr.s_check1), tf.shape(ptr.i_check1)], feed_dict={input_: input_batch}))
            # print('ss', ss)
            # print('ll', ll)
            print(sess.run([tf.shape(graphs_gen), tf.shape(reward_probs)], feed_dict={input_: input_batch}))


    # self.samples is seq_lenthg * batch size * seq_length