import tensorflow as tf
class nn(object):

    save_period = 50
    save_path = 'q_model/model'

    def __init__(self, state_shape, layers, hidden_units):
        self.layers = layers
        self.hidden_units = hidden_units
        self.num_classes = 1

        #@todo: Add regularization
        #Model
        self.inputs = tf.placeholder(dtype= tf.float32, shape=[None] + state_shape)
        # self.action = tf.placeholder(dtype= tf.float32, shape=[None, 1])
        self.targets = tf.placeholder(dtype=tf.float32, shape=[None])
        self.learning_rate = tf.placeholder(dtype=tf.float32)
        self.dropout = tf.placeholder(dtype=tf.float32)

        #layer1
        W1 = tf.Variable(tf.truncated_normal(shape=state_shape + [self.hidden_units], stddev= 0.01))
        B1 = tf.Variable(tf.ones(shape=[self.hidden_units]))
        self.o1 = tf.matmul(self.inputs, W1)+B1
        self.o_act_1 = tf.nn.relu(self.o1)

        self.o_act_1 = tf.nn.dropout(self.o_act_1, keep_prob= self.dropout)

        #layer2
        W2 = tf.Variable(tf.truncated_normal(shape= [self.hidden_units, self.num_classes], stddev= 0.01))
        B2 = tf.Variable(tf.ones([self.num_classes]))
        self.o2 = tf.matmul(self.o_act_1, W2)+B2

        self.o2_1 = tf.reshape(self.o2, shape= [-1])

        # self.o_act_2 = tf.nn.softmax(self.o2_drop)

        # self.prediction = tf.arg_max(self.o_act_2, dimension=0)
        self.prediction = self.o2_1
        self.loss = tf.reduce_mean(tf.square( self.o2_1-self.targets))

        self.opt = tf.train.AdamOptimizer(learning_rate= self.learning_rate).minimize(self.loss)

        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def train_model(self, data, learning_rate, num_epochs,batch_size = None):
        feed_dict = {self.inputs : data[0],
                     self.targets : data[1],
                     self.learning_rate : learning_rate,
                     self.dropout: 0.5}

        for epoch in range(num_epochs):
            _, loss = self.session.run([self.opt, self.loss], feed_dict= feed_dict)
            print "<< Q >> epoch : "+str(epoch)+" loss : "+str(loss)

            if epoch%self.save_period == 0:
                self.saver.save(sess= self.session,
                                save_path= self.save_path)

    def restore(self):
        # load latest model
        # model_path = self.save_path
        model_path = '/home/mvidyasa/Desktop/gym/examples/my_agents/q_model/model'
        self.saver.restore(self.session, save_path=model_path)

    def get_q_value(self, input):

        # input = tf.expand_dims(input, axis=0)
        feed_dict = {self.inputs : input,
                     self.dropout: 1.}

        prediction = self.session.run([self.prediction],feed_dict)
        # if prediction > 0.5:
        #     prediction = 1
        # else:
        #     prediction = 0
        return prediction