import os
import cv2
import glob
from scipy import ndimage
import glob
import shutil
import random

folder_name = ["OK", "NG"]
Imgsize = (250, 250)

for i in folder_name:
    print("{}の写真を増やします".format(i))
    in_dir = "image/original/" + i + "/*"
    out_dir = "image/edited/" + i
    os.makedirs(out_dir, exist_ok=True)
    in_jpg=glob.glob(in_dir)
    
    for x in range(len(in_jpg)):
        img = cv2.imread(str(in_jpg[x]))
        for ang in [-10, 0, 10]:
            img_rot = ndimage.rotate(img, ang)
            img_rot = cv2.resize(img_rot, Imgsize)
            fileName = os.path.join(out_dir, str(x)+"_"+str(ang)+".jpg")
            cv2.imwrite(str(fileName), img_rot)

            img_filter = cv2.GaussianBlur(img_rot, (5, 5), 0)
            fileName = os.path.join(out_dir, str(ｘ)+"_"+str(ang)+"filter.jpg")
            cv2.imwrite(str(fileName), img_filter)


print("completed1")

for name in folder_name:
    in_dir = "image/edited/"+name+"/*"
    in_jpg = glob.glob(in_dir)
    

    random.shuffle(in_jpg)
    os.makedirs("image/test/"+name, exist_ok=True)
    for t in range(len(in_jpg)//5):
        shutil.move(str(in_jpg[t]), "image/test/"+name)

print("completed2")

import tensorflow as tf
from keras.utils.np_utils import to_categorical
import os
import numpy as np
import cv2

#----------------------------------------------------------------------------------------------

X_train = []
Y_train = []
for i in range(len(folder_name)):
    img_file_name_list = os.listdir("image/edited/"+ folder_name[i])
    
    for j in range(0, len(img_file_name_list)-1):
        n = os.path.join("image/edited/"+folder_name[i]+"/", img_file_name_list[j])
        img = cv2.imread(n)
        if img is None:
            print("image"+str(j)+":NoImage1")
            continue
        else:
            r,g,b = cv2.split(img)
            img = cv2.merge([r,g,b])
            X_train.append(img)
            Y_train.append(i)
            
print("completed3")

import cv2

X_test = []
Y_test = []
for i in range(len(folder_name)):
    img_file_name_list = os.listdir("image/test/"+folder_name[i])
    
    for j in range(0, len(img_file_name_list)-1):
        n = os.path.join("image/test/"+folder_name[i]+"/", img_file_name_list[i])
        img = cv2.imread(n)
        if img is None:
            print("image"+str(j)+":Noimage2")
            continue
        else:
            r, g, b = cv2.split(img)
            img = cv2.merge([r,g,b])
            X_test.append(img)
            Y_test.append(i)
print("completed4")
            
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = to_categorical(Y_train, 2)
y_test = to_categorical(Y_test, 2)

print("completed")

from tensorflow.keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
input_shape = (250, 250, 3)

model = Sequential()
model.add(Conv2D(input_shape=input_shape, filters=32, kernel_size=(3, 3), 
                strides=(1, 1), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(3, 3),
                strides=(1, 1), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation("sigmoid"))
model.add(Dense(128))
model.add(Activation("sigmoid"))
model.add(Dense(2))
model.add(Activation ("softmax"))

model.compile(optimizer="sgd",
             loss="categorical_crossentropy",
             metrics=["accuracy"])

history = model.fit(X_train, y_train, batch_size=10, 
                   epochs=20, verbose=1, validation_data=(X_test, y_test))

model.save("final_model.h5")

plt.plot(history.history["accuracy"], label="acc", ls="-", marker="o")
plt.plot(history.history["val_accuracy"], label="val_acc", ls="-", marker="x")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(loc="best")
plt.show()
print("completed5")

            
            