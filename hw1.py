from PIL import Image
from os.path import isfile, isdir, join, split, splitext, exists
from os import listdir, makedirs
import numpy as np

'''
read file
'''
path = 'CroppedYale/'

directories = [join(path, d)
               for d in listdir(path) 
               if isdir(join(path, d))]

# 35 training data
training_data = [join(d,f)
                    for d in directories
                    for i, f in enumerate(listdir(d))
                    if isfile(join(d, f)) and i < 35 and splitext(f)[-1] != ".bad"
            ]

# 30 testing data
testing_data = [join(d,f)
                    for d in directories
                    for i, f in enumerate(listdir(d))
                    if isfile(join(d, f)) and i >= 35 and splitext(f)[-1] != ".bad"
            ]


'''
start testing
'''

SADPrediction = [0] * len(testing_data)                             # save the predictions of each testing image for SAD
SSDPrediction = [0] * len(testing_data)                             # save the predictions of each testing image for SSD

for i, test in enumerate(testing_data):
    min_SAD_value = 9999999999999
    min_SSD_value = 9999999999999
    min_SAD_name = None
    min_SSD_name = None
    im_test = Image.open(test).convert("L")                         # read the testing image and convert it to grey scale                            
    
    for train in training_data:
        im_train = Image.open(train).convert("L")                   # read the training image and convert it to grey scale
        
        x = np.array(im_test,dtype=np.int64)
        y = np.array(im_train,dtype=np.int64)
        #print(x.shape)
        #print(y.shape)
        z = abs(x/100 - y/100)
        z = z*100

        SAD = np.sum(z)                                             # calculate SAD and save the file name
        if SAD < min_SAD_value:
            min_SAD_value = SAD
            min_SAD_name = test
            print(min_SAD_name,"*******",min_SAD_value)
            SADPrediction[i] = min_SAD_name
            
        SSD = np.sum(z**2)                                          # calculate SSD and save the file name
        if SSD < min_SSD_value:
            min_SSD_value = SSD
            min_SSD_name = test
            print(min_SSD_name,"*******",min_SSD_value)
            SSDPrediction[i] = min_SSD_name



#print(im.format, im.size, im.mode)