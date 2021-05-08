Steps for training:
# Step 0:
Install all required libraries by running command: pip install -r requirements-pip.txt
# Step 1: 
Create 2 empty folders under this repository, named "datasets" and "results"
# Step 2: 
Download dataset and extract under this repository 
After step 1 and step 2, we have these directory:

![image](https://user-images.githubusercontent.com/67949536/117542287-35179480-b042-11eb-9d2c-494b1fde00aa.png)


The directory "Pneumonia_NORMAL_128x128" is a directory containing images of the same size (IMG_SIZE, IMG_SIZE), and IMG_SIZE has the form 2^a, for example (128,128). 
And example of data can be downloaded from here: 
https://drive.google.com/file/d/114gKucljINsAZPI0p1nkqUibSB2Os400/view?usp=sharing
This dataset contains images of size (128,128) and has the class "NORMAL". 

After extracting, we have this structure: 


![image](https://user-images.githubusercontent.com/67949536/117542414-c2f37f80-b042-11eb-8f36-c9ddf8a36698.png)



# Step 3: 
Prepare dataset for training by running command: 
python dataset_tool.py create_from_images datasets/pneumonia Pneumonia_NORMAL_128x128 
After this, it would create some tf-records file into the folder "datasets/pneumonia".
Like this: 

![image](https://user-images.githubusercontent.com/67949536/117542499-16fe6400-b043-11eb-9981-d23a27ff3986.png)


These files would be loaded and used for training model 

# Step 4: Modify file config.py
This file contains all configures for our training
Some important variables to modify:
- The variable "data_dir" and "result_dir" in line 21, 22 
- The line 53 means that our tf-records files are under the directory "pneumonia"
- From line 128-139, we need to uncomment exactly one line, and then specify how many GPUS we will use. (This is important because more GPUS would speed up the training).

The more detailed guidance can be found in the "Training networks" section in the original project link: https://github.com/tkarras/progressive_growing_of_gans

# Step 5: Train model 
Run the file train.py 

# Notice
This Progressive Grow GAN will learn to generate images that have the same size as our original dataset, for example if we use the dataset Pneumonia_NORMAL_128x128 as above, then it will learn to generate images of size (128,128). 
