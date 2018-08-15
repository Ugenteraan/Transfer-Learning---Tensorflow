import os
import sys
import glob
from random import shuffle

#to keep track of files
number_of_files = 0

#get the current terminal's path
current_directory = os.getcwd()

#path where the datasets are stored
dataset_path = current_directory + '/datasets/'


#transfer learning mode
#refer to documentation
transfer_learning_mode_array = ['first', 'second', 'third']
transfer_learning_mode = transfer_learning_mode_array[1]

#when mode #2 is picked, you must provide how many cnn layers' weights you want to freeze (starts from 1 and ends with 16)

no_of_layers_to_freeze_start = 1
no_of_layers_to_freeze_end = 9

#if the transfer learning mode is first
#freeze all the conv layers
if transfer_learning_mode == 'first':
	no_of_layers_to_freeze_start = 1
	no_of_layers_to_freeze_end = 13

if transfer_learning_mode == 'third':
	no_of_layers_to_freeze_start = 0
	no_of_layers_to_freeze_end = 0

#errorneous values for the second mode
if transfer_learning_mode == 'second' :
	#vgg-16 only has 13 conv layers (including fc we have 16 layers)
	if no_of_layers_to_freeze_start == None or no_of_layers_to_freeze_end > 16:

		print("ERROR in no_of_layers_to_freeze!")
		sys.exit()




#this list keeps track of which weight can be trained and which cant be trained
trainables = []

for k in range(no_of_layers_to_freeze_start-1, no_of_layers_to_freeze_end):
	#freeze these layers
	trainables.append(False)

list_len = len(trainables)

for j in range(list_len-1, 16):
	#rest of the layers are trainable
	trainables.append(True)


print(trainables)

#get the length to extract the name of the folder (i.e. styles) later on
dataset_path_length = len(dataset_path)

#determine whether the input images are coloured or grayscaled
grayscale = False

#set the depth of the image (grayscale = 1 channel only, coloured = 3 channels)
image_depth = 3 if grayscale is False else 1

#size of the image (height and width)
picture_input_dimension = 224

#training iteration
epoch = 2000

#this is to break the datasets in 'n' sizes 
memory_limit = 1

#stochastic gradient descent
batch_size = 30
accuracy_batch_size = 30

#learning rate of the model
learning_rate = 1e-5

#names of the folder to save and load models
save_folder_name = 'transferred_ckpt'
load_folder_name = 'saved_ckpt'

#in list format in case there are other image extensions in the future
image_extension = ['.jpg']

#list to keep track of the name of the folder (i.e. styles)
folders = []

#names of all the files
filenames = []

#function to check the extension of the file using the 'image_extension' variable earlier
def check_file_type(file_path):
	#one-liner code that returns true if the file's extension ends with '.jpg', returns false if not
	return True if file_path[-4:] in image_extension else False
	


#for all the folders (i.e. styles) in the directory, append to the list
for folder in glob.glob(dataset_path + '*'):
	#append the folder's name into the list
	#this is where the dataset_path_length variable declared earlier is used
	folders.append(folder[dataset_path_length:])


#sort the list alphabetically
folders = sorted(folders, key=str.lower)


#to iterate through all the data files and keep count of the number of files
folder_count = 0
for folder in folders:

	for file in glob.glob(dataset_path + folder + '/**', recursive=True):

		#only true if the file is indeed a file that ends with a '.jpg' extension
		if check_file_type(file):
			#increase the count of the variable everytime there's an image file
			number_of_files += 1
			filenames.append([file, folder_count])

	folder_count += 1

#randomize the list
shuffle(filenames)

print("CLASSES : ", folders)
#number of classes = number of folders
num_of_classes = len(folders)

print("TOTAL NUMBER OF FILES : ", number_of_files)