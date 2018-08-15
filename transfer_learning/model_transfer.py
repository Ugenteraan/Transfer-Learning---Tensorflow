#VGGNET16 Neural Network

import settings
import tensorflow as tf 
import collections

class Model:

	


	def __init__(self):

		conv_trainable = settings.trainables
		image_dimension = settings.picture_input_dimension
		image_depth = settings.image_depth
		filter_size1, filter_size2, filter_size3, filter_size4 = 3, 3, 3, 3
		patch_depth1, patch_depth2, patch_depth3, patch_depth4 = 64, 128, 256, 512
		num_of_hidden_layer1, num_of_hidden_layer2 = 4096, 1024

		#trainable paramater controls if the weight should be freezed or not
		#if trainable is set to True, then the weight is NOT FREEZED (means it is trainable)
		w1 = tf.get_variable('w1', shape=[filter_size1, filter_size1, image_depth, patch_depth1], trainable=conv_trainable[0], initializer=tf.truncated_normal_initializer(stddev = 0.1))
		b1 = tf.get_variable('b1', shape=[patch_depth1], trainable=conv_trainable[0], initializer=tf.truncated_normal_initializer(stddev = 0.3))
		w2 = tf.get_variable('w2', shape=[filter_size1, filter_size1, patch_depth1, patch_depth1], trainable=conv_trainable[1], initializer=tf.truncated_normal_initializer(stddev = 0.1))
		b2 = tf.get_variable('b2', shape=[patch_depth1], trainable=conv_trainable[1], initializer=tf.truncated_normal_initializer(stddev = 0.1))

		w3 = tf.get_variable('w3', shape=[filter_size2, filter_size2, patch_depth1, patch_depth2], trainable=conv_trainable[2], initializer=tf.truncated_normal_initializer(stddev = 0.1))
		b3 = tf.get_variable('b3', shape=[patch_depth2], trainable=conv_trainable[2], initializer=tf.truncated_normal_initializer(stddev = 0.1))
		w4 = tf.get_variable('w4', shape=[filter_size2, filter_size2, patch_depth2, patch_depth2], trainable=conv_trainable[3], initializer=tf.truncated_normal_initializer(stddev = 0.1))
		b4 = tf.get_variable('b4', shape=[patch_depth2], trainable=conv_trainable[3], initializer=tf.truncated_normal_initializer(stddev = 0.1))

		w5 = tf.get_variable('w5', shape=[filter_size3, filter_size3, patch_depth2, patch_depth3], trainable=conv_trainable[4], initializer=tf.truncated_normal_initializer(stddev = 0.1))
		b5 = tf.get_variable('b5', shape=[patch_depth3], trainable=conv_trainable[4], initializer=tf.truncated_normal_initializer(stddev = 0.1))
		w6 = tf.get_variable('w6', shape=[filter_size3, filter_size3, patch_depth3, patch_depth3], trainable=conv_trainable[5], initializer=tf.truncated_normal_initializer(stddev = 0.1))
		b6 = tf.get_variable('b6', shape=[patch_depth3], trainable=conv_trainable[5], initializer=tf.truncated_normal_initializer(stddev = 0.1))
		w7 = tf.get_variable('w7', shape=[filter_size3, filter_size3, patch_depth3, patch_depth3], trainable=conv_trainable[6], initializer=tf.truncated_normal_initializer(stddev = 0.1))
		b7 = tf.get_variable('b7', shape=[patch_depth3], trainable=conv_trainable[6], initializer=tf.truncated_normal_initializer(stddev = 0.1))

		w8 = tf.get_variable('w8', shape=[filter_size4, filter_size4, patch_depth3, patch_depth4], trainable=conv_trainable[7], initializer=tf.truncated_normal_initializer(stddev = 0.1))
		b8 = tf.get_variable('b8', shape=[patch_depth4], trainable=conv_trainable[7], initializer=tf.truncated_normal_initializer(stddev = 0.1))
		w9 = tf.get_variable('w9', shape=[filter_size4, filter_size4, patch_depth4, patch_depth4], trainable=conv_trainable[8], initializer=tf.truncated_normal_initializer(stddev = 0.1))
		b9 = tf.get_variable('b9', shape=[patch_depth4], trainable=conv_trainable[8], initializer=tf.truncated_normal_initializer(stddev = 0.1))
		w10 = tf.get_variable('w10', shape=[filter_size4, filter_size4, patch_depth4, patch_depth4], trainable=conv_trainable[9], initializer=tf.truncated_normal_initializer(stddev = 0.1))
		b10 = tf.get_variable('b10', shape=[patch_depth4], trainable=conv_trainable[9], initializer=tf.truncated_normal_initializer(stddev = 0.1))

		w11 = tf.get_variable('w11', shape=[filter_size4, filter_size4, patch_depth4, patch_depth4], trainable=conv_trainable[10], initializer=tf.truncated_normal_initializer(stddev = 0.1))
		b11 = tf.get_variable('b11', shape=[patch_depth4], trainable=conv_trainable[10], initializer=tf.truncated_normal_initializer(stddev = 0.1))
		w12 = tf.get_variable('w12', shape=[filter_size4, filter_size4, patch_depth4, patch_depth4], trainable=conv_trainable[11], initializer=tf.truncated_normal_initializer(stddev = 0.1))
		b12 = tf.get_variable('b12', shape=[patch_depth4], trainable=conv_trainable[11], initializer=tf.truncated_normal_initializer(stddev = 0.1))
		w13 = tf.get_variable('w13', shape=[filter_size4, filter_size4, patch_depth4, patch_depth4], trainable=conv_trainable[12], initializer=tf.truncated_normal_initializer(stddev = 0.1))
		b13 = tf.get_variable('b13', shape=[patch_depth4], trainable=conv_trainable[12], initializer=tf.truncated_normal_initializer(stddev = 0.1))  

		no_of_pooling_layer = 5

		w14 = tf.Variable(tf.truncated_normal([((image_dimension // (2**no_of_pooling_layer)) * (image_dimension // (2**no_of_pooling_layer))*patch_depth4), num_of_hidden_layer1], stddev=0.1), name='w14', trainable=conv_trainable[13])
		b14 = tf.Variable(tf.constant(1.0, shape = [num_of_hidden_layer1]), name='b14', trainable=conv_trainable[13])

		w15 = tf.Variable(tf.truncated_normal([num_of_hidden_layer1, num_of_hidden_layer2], stddev=0.1), name='w15', trainable=conv_trainable[14])
		b15 = tf.Variable(tf.constant(1.0, shape = [num_of_hidden_layer2]), name='b15', trainable=conv_trainable[14])

		w16 = tf.Variable(tf.truncated_normal([num_of_hidden_layer2, settings.num_of_classes], stddev=0.1), name='w16', trainable=conv_trainable[15])
		b16 = tf.Variable(tf.constant(1.0, shape = [settings.num_of_classes]), name='b16', trainable=conv_trainable[15])

		#must use ordered collections. Normal dictionary does not keep its order
		self.variables = collections.OrderedDict(
			[('w1',w1), ('b1', b1), ('w2',w2), ('b2',b2), ('w3', w3), ('b3', b3), ('w4', w4), ('b4', b4), ('w5', w5), ('b5', b5), ('w6', w6), ('b6', b6), ('w7', w7), ('b7', b7), ('w8', w8), ('b8', b8),
			 ('w9', w9), ('b9',b9), ('w10', w10), ('b10', b10), ('w11', w11), ('b11', b11), ('w12', w12), ('b12', b12), ('w13', w13), ('b13', b13), ('w14', w14), ('b14', b14), ('w15', w15), ('b15', b15), 
			 ('w16', w16), ('b16', b16)]
			)
		#to iterate
		key_list = ['w1','b1','w2','b2','w3','b3','w4','b4','w5','b5','w6','b6','w7','b7','w8','b8','w9','b9','w10','b10','w11','b11','w12','b12','w13','b13','w14','b14','w15','b15','w16','b16']

		#to load the weights from the ckpt file
		self.weights_to_load = {}

		#iterates from the desired number of layers to freeze - 1 ( minus 1 because index starts from 0) until the number of end layer to freeze * 2 (*2 because of the bias variable)
		for i in range(settings.no_of_layers_to_freeze_start -1 , settings.no_of_layers_to_freeze_end*2):

			#append the dictionary
			self.weights_to_load[key_list[i]] = self.variables[key_list[i]]

		
		
		self.x = tf.placeholder('float', [None, image_dimension, image_dimension , image_depth], name='x')   
		
		self.y_= tf.placeholder('float', [None, settings.num_of_classes], name='y_')

		self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

		x_image = tf.reshape(self.x , [-1, image_dimension, image_dimension, 3])

		with tf.name_scope('Conv-group1'):

			layer1_conv = tf.nn.conv2d(x_image, self.variables['w1'], strides=[1,1,1,1], padding='SAME')
			layer1_actv = tf.nn.relu(layer1_conv + self.variables['b1'])
			layer2_conv = tf.nn.conv2d(layer1_actv, self.variables['w2'], strides=[1,1,1,1], padding='SAME')
			layer2_actv = tf.nn.relu(layer2_conv + self.variables['b2'])

		with tf.name_scope('Conv1-maxPooling'):
			layer2_pool = tf.nn.max_pool(layer2_actv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

		with tf.name_scope('Conv-group2'):

			layer3_conv = tf.nn.conv2d(layer2_pool, self.variables['w3'], strides=[1,1,1,1], padding='SAME')
			layer3_actv = tf.nn.relu(layer3_conv + self.variables['b3'])
			layer4_conv = tf.nn.conv2d(layer3_actv, self.variables['w4'], strides=[1,1,1,1], padding='SAME')
			layer4_actv = tf.nn.relu(layer4_conv + self.variables['b4'])

		with tf.name_scope('Conv2-maxPooling'):

			layer4_pool = tf.nn.max_pool(layer4_actv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

		with tf.name_scope('Conv-group3'):

			layer5_conv = tf.nn.conv2d(layer4_pool, self.variables['w5'], strides=[1,1,1,1], padding='SAME')
			layer5_actv = tf.nn.relu(layer5_conv + self.variables['b5'])
			layer6_conv = tf.nn.conv2d(layer5_actv, self.variables['w6'], strides=[1,1,1,1], padding='SAME')
			layer6_actv = tf.nn.relu(layer6_conv + self.variables['b6'])
			layer7_conv = tf.nn.conv2d(layer5_actv, self.variables['w7'], strides=[1,1,1,1], padding='SAME')
			layer7_actv = tf.nn.relu(layer7_conv + self.variables['b7'])

		with tf.name_scope('Conv3-maxPooling'):

			layer7_pool = tf.nn.max_pool(layer7_actv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

		with tf.name_scope('Conv-group4'):

			layer8_conv = tf.nn.conv2d(layer7_pool, self.variables['w8'], strides=[1,1,1,1], padding='SAME')
			layer8_actv = tf.nn.relu(layer8_conv + self.variables['b8'])
			layer9_conv = tf.nn.conv2d(layer8_actv, self.variables['w9'], strides=[1,1,1,1], padding='SAME')
			layer9_actv = tf.nn.relu(layer9_conv + self.variables['b9'])
			layer10_conv = tf.nn.conv2d(layer8_actv, self.variables['w10'], strides=[1,1,1,1], padding='SAME')
			layer10_actv = tf.nn.relu(layer10_conv + self.variables['b10'])

		with tf.name_scope('Conv4-maxPooling'):

			layer10_pool = tf.nn.max_pool(layer10_actv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

		with tf.name_scope('Conv-group5'):

			layer11_conv = tf.nn.conv2d(layer10_pool, self.variables['w11'], strides=[1,1,1,1], padding='SAME')
			layer11_actv = tf.nn.relu(layer11_conv + self.variables['b11'])
			layer12_conv = tf.nn.conv2d(layer11_actv, self.variables['w12'], strides=[1,1,1,1], padding='SAME')
			layer12_actv = tf.nn.relu(layer12_conv + self.variables['b12'])
			layer13_conv = tf.nn.conv2d(layer11_actv, self.variables['w13'], strides=[1,1,1,1], padding='SAME')
			layer13_actv = tf.nn.relu(layer13_conv + self.variables['b13'])

		with tf.name_scope('Conv5-maxPooling'):

			layer13_pool = tf.nn.max_pool(layer13_actv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

		with tf.name_scope('Flatten-layer'):

			output_shape = (image_dimension // (2**no_of_pooling_layer)) * (image_dimension // (2**no_of_pooling_layer))*patch_depth4

			self.reshape_output = tf.reshape(layer13_pool, [-1, output_shape])

			
		with tf.name_scope('FC_layers'):

			fc_1_layer = tf.add(tf.matmul(self.reshape_output, self.variables['w14']), self.variables['b14'])
			fc_1_actv = tf.nn.relu(fc_1_layer)

			with tf.name_scope('Dropout-fc1'):

				dropout_layer1 = tf.nn.dropout(fc_1_actv, self.keep_prob)

			fc_2_layer = tf.add(tf.matmul(dropout_layer1, self.variables['w15']), self.variables['b15'])
			fc_2_actv = tf.nn.relu(fc_2_layer)

			with tf.name_scope('Dropout-fc2'):
# 
				dropout_layer2 = tf.nn.dropout(fc_2_actv, self.keep_prob)


		with tf.name_scope('Logits'):

			self.logits = tf.add(tf.matmul(dropout_layer2, self.variables['w16']), self.variables['b16'], name='logits')

		with tf.name_scope('Prediction'):

			self.y = tf.nn.softmax(self.logits, name='y')

		with tf.name_scope("Training"):

			regularization_value = 0.01 * tf.nn.l2_loss(self.variables['w14']) + 0.01 * tf.nn.l2_loss(self.variables['w15']) + 0.01 * tf.nn.l2_loss(self.variables['w16'])

			self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.logits) + regularization_value)
			

			self.train_step = tf.train.AdamOptimizer(learning_rate= settings.learning_rate).minimize(self.cost)
		

		with tf.name_scope("Accuracy"):

			correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y_, 1))

			with tf.name_scope("Accuracy-2"):

				self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
				
		#add these to collections so that it can be accessed from the meta graph later.
		tf.add_to_collection("logits", self.logits)
		tf.add_to_collection("x", self.x)
		tf.add_to_collection("y", self.y)
		tf.add_to_collection("keep_prob", self.keep_prob)
		tf.add_to_collection("y_", self.y_)
		tf.add_to_collection("accuracy", self.accuracy)
		