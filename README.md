# TRANSFER LEARNING WITH TENSORFLOW


##  Requirements
* Tensorflow == 1.9.0
* Python 3.5
* OpenCV == 4.0.0-pre

##  Usage

>NOTE : This transfer learning is performed using CNN's VGG-16 architecture. If you want to modify the architecture, do so in **standard_training > model.py** and **transfer_learning > model_transfer.py**.  

###  Instructions
1) Perform a standard image recognition and classification training.  
1.1) Create a folder called **datasets** in **standard_training** folder and place the image dataset that are in their own corresponding folders ( which is their label ) inside the **datasets** folder that you just created.  
1.2) Change the hyper-parameters as you wish in _settings.py_.  
1.3) Run _main.py_ file to start the training.  
2) Perform transfer learning.  
2.1) Copy all the files from **standard_training > ckpt_folder** folder into **transfer_learning > saved_ckpt** folder.  
2.2) Create a folder called **datasets** in **transfer_learning** folder and place the image dataset that are in their own corresponding folders ( which is their label ) inside the **datasets** folder that you just created. _Important NOTE : Since this is a transfer learning, you need to have all the dataset classes that you have used to perform the standard training ( in a little amount ) **PLUS** the new class(es) that you're going to add into the model's prediction classes. However, take note of the amount of the images that you're using. It should be balanced!_  
2.3) Change the hyper-parameters as you wish in _settings.py_.  
2.4) Notice that there are three types of transfer learning. First choice only replaces the FC layers. Second choice replaces FC layers along with the defined number of Convolution layers from the back. Lastly, the third choice replaces none of the layer ( it loads the weights as a checkpoint and continue the training provided the dataset classes are the same!)   
3) Validation.  
3.1) Validation can be performed using the _validation.py_ files in both the folders. Make sure the `dataset_path` variable in _settings.py_ has been changed to the validation dataset folder.  
3.2) If you're performing validation on the model that has undergone the transfer learning process, make sure to change the `load_folder_name` to `transferred-ckpt`.
