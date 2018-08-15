import tensorflow as tf 
import numpy as np 
import cv2
import os
import glob
from sklearn.model_selection import train_test_split
import settings
import utils

#reset all tensorflow graphs
tf.reset_default_graph()

#initialize tensorflow graphs
g1 = tf.Graph()

#import meta file for graph 1
with g1.as_default():

	saver_g1 = tf.train.import_meta_graph(os.getcwd() + '/datasets/model.ckpt.meta')



#creates one session for each graphs
sess1 = tf.Session(graph=g1)

#load the checkpoint file for each session
try:
	saver_g1.restore(sess1, os.getcwd()+'/datasets/model.ckpt')
	# saver_g1.restore(sess2, os.getcwd()+'/3-layer-vgg/ckpt_folder/model.ckpt')
	print("Models have been loaded!")
except:
	print("Errors loading models!")

#get the variables from graph1 and graph2
graph1_logit = g1.get_collection('logits')[0]
graph1_x = g1.get_collection('x')[0]
graph1_keep_prob = g1.get_collection('keep_prob')[0]
graph1_y_ = g1.get_collection('y_')[0]
graph1_accuracy = g1.get_collection('accuracy')[0]

#initialize variables for each graphs
graph1_dimension = 256
graph1_image_depth = 3




X, X_test, Y, Y_test = utils.get_files(1)

X = np.concatenate((X,X_test), axis=0)
Y = np.concatenate((Y, Y_test), axis=0)

print(X.shape, Y.shape)


one_hot_vector = np.zeros((Y.shape[0], settings.num_of_classes))


#change the labels into one hot vector
for i in range(Y.shape[0]):

	one_hot_vector[i][int(Y[i])] = 1.0



#function to perform the testing session
def testing_session():

	#variable to keep count how many times a certain size of batch has went into the loop to complete an epoch
	testing_counter = 0
	#to keep the sum of the testing accuracies
	test_accuracy = 0
	#loop through the dataset with the specified step (batch size)
	for index in range(0, X.shape[0], settings.accuracy_batch_size):
		#assign the last index of the batch to the variable
		end_batch_size_accuracy = index + settings.accuracy_batch_size
		#if the assigned index number is out of the bound, then it will be set to None 
		if end_batch_size_accuracy >= X.shape[0] : end_batch_size_accuracy = None 
		#perform the testing with the specified batch size
		test_accuracy += graph1_accuracy.eval(session=sess1, feed_dict={graph1_x : X[index : end_batch_size_accuracy], graph1_y_: one_hot_vector[index : end_batch_size_accuracy],  graph1_keep_prob:1.0})
		#increase the countergraph1
		testing_counter += 1

	return(test_accuracy/testing_counter)
#resize the image according to graph1's settings
#get the prediction from graph1

print(testing_session())

# result = sess1.run(graph1_logit, feed_dict={graph1_x: input_image_graph1, graph1_keep_prob:1.0})
# print("Graph 1 prediction : " ,graph1_classes[np.argmax(result)])

