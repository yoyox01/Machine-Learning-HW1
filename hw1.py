from PIL import Image
from os.path import isfile, isdir, join, split, splitext, exists
from os import listdir, makedirs
import numpy as np

'''
defines
'''
def calculate_SAD(arr1,arr2):
    z = abs(arr1/100 - arr2/ 100)
    z = z * 100
    return np.sum(z)

def calculate_SSD(arr1,arr2):
    z = abs(arr1/100 - arr2/100)
    z = z * 100
    return np.sum(z**2)



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
sad_predict = {}                                                    # save the predictions of each testing image for SAD
ssd_predict = {}                                                    # save the predictions of each testing image for SSD

for i, test in enumerate(testing_data):
    min_SAD_value = 9999999999999
    min_SSD_value = 9999999999999
    min_SAD_name = None
    min_SSD_name = None
    im_test = Image.open(test).convert("L")                         # read the testing image and convert it to grey scale                            
    
    sad_predict[test] = {}
    ssd_predict[test] = {}
    for train in training_data:
        im_train = Image.open(train).convert("L")                   # read the training image and convert it to grey scale
        
        x = np.array(im_test,dtype=np.int64)
        y = np.array(im_train,dtype=np.int64)

        SAD = calculate_SAD(x, y)                                   # calculate SAD and save the file name
        if SAD < min_SAD_value:
            min_SAD_value = SAD
            min_SAD_name = test
            print(min_SAD_name,"*******",min_SAD_value)
            sad_predict[test] = {train: min_SAD_value}
            
        SSD = calculate_SSD(x, y)                                   # calculate SSD and save the file name
        if SSD < min_SSD_value:
            min_SSD_value = SSD
            min_SSD_name = test
            print(min_SSD_name,"*******",min_SSD_value)
            ssd_predict[test] = {train: min_SSD_value}

'''
accuracy
'''
correct_sad = 0
for k in sad_predict.keys():
    for pk in sad_predict[k].keys():
        if k.replace('\\','/').split('/')[1] in pk:
            correct += 1
accuracy_sad = correct_sad / len(testing_data)
print("SAD :\n\tcorrect : {0} images in {1} testing data \n\taccuracy : {1}".format(correct_sad,len(testing_data),accuracy_sad))

correct_ssd = 0
for k in ssd_predict.keys():
    for pk in ssd_predict[k].keys():
        if k.replace('\\','/').split('/')[1] in pk:
            correct += 1
accuracy_ssd = correct_ssd / len(testing_data)
print("SSD :\n\tcorrect : {0} images in {1} testing data \n\taccuracy : {1}".format(correct_ssd,len(testing_data),accuracy_ssd))

