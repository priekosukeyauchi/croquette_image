import os
import glob

ok = "image/original/OK/*"
ng = "image/original/NG/*"

fol_list = [ok, ng]
class_list = ["OK", "NG"]

n=0

for folder_list in fol_list:    
    flist = glob.glob(folder_list)
    print(flist)
    print(len(flist))
    i = 1
    for image in flist:
        os.rename(image, "image/original/" + class_list[n]+ "/image_" + str(i) + ".jpg")
        i += 1
    n += 1