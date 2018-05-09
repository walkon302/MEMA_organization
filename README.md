# MEMA_organization
Authors: Chun-Han Lin, Tiina Jokela  
Email: walkon302@gmail.com  
Last Update: 5/9/2018    

# Purpose and the use cases

MEMA can be used to study impacts of microenvironment on many cellular responses, including cellular morphological changes, drug responses, and cellular organization. MEMA images are processed either by CellProfiler or Python and generate the spreadsheet containing all the measurements. While most of the measurements are single-cell level data, which can be analyzed directly; cellular organization patterns are multiple-cells level information, which cannot be readily included in the result of the MEMA images analysis.

It takes time to manually classify the MEMA images to different organization patterns and label each image with either well-organized or disorganized. Thus, the goal of this project is to develope a method to automate the process of labeling images with organization patterns for facilitating further MEMA analysis.

This automation can be intergrated into MEMA analysis pipeline or be used individually to access the cellular organization patterns in other experiments.

### Use case 1: For the MEMA experiments
After applying this method, the MEMA images are classified into different organization patterns. We can then use this information to study which microenvironmental components play important roles in cellular organization.

### Use case 2: For other experiments
After applying this method, we will be able to quicktly quantify the ratio of organization patterns in the experiment.

# How to use

* Open terminal, the following procedures are all in the terminal

#### Type in the following commands in the terminal directly
* install anaconda https://repo.anaconda.com/archive/Anaconda2-5.1.0-MacOSX-x86_64.pkg  
-- install the anaconda  

* conda create -n mema python=2.7 anaconda  
-- create a new virtual environment called mema for running the script without altering the original system  

* source activate mema  
-- activate the virtual environment  

* conda install -c https://conda.anaconda.org/menpo opencv3  
-- install proper version of opencv  

* pip install grpcio==1.9.1 https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.6.0-py2-none-any.whl  
-- install proper version of tensorflow  

#### Ready for using the script
* https://github.com/walkon302/MEMA_organization/archive/master.zip  
-- download the file and unzip it on the desktop. Rename it for a shorter name, e.g., 'mema' since the original name 'MEMA_organization-master' is too long.

#### Type in the commands in the terminal directly
* cd ./desktop/mema  
-- enter the main folder of the script  

* mkdir input  
-- generate a folder named input  

* cd input  
-- enter the subfolder, input  

* mkdir ori_organized  
* mkdir ori_disorganized  
* mkdir predict
-- generate two folders, one is for organized, another one is for disorganized, the last one is for new images that need to be classified.  

#### Then move images and model into this mema folder
* Put images into ori_organized and ori_disorganized folders accordingly.  
* Put the images that need to be classified into predict folder.  
* Put MEMA_model into main folder, mema.  

#### Type in the commands
* cd..  
-- go back to main folder, mema  

* cd src  
-- enter src folder  


#### Ready to use, type in the commands.
* python main.py train  
-- Train the model with images from ori_organized and ori_disorganized folders  

* python main.py eval  
-- Evaluate the performace  

* pyton main.py predict  
-- Classify the images in the predict folder and output a output.csv file in the output folder.  
