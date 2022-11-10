import cv2
from os import listdir
from os.path import isfile,join
mypath='./videos_vis/'
files=[f for f in listdir(mypath) if isfile(join(mypath,f))]
print(files)
files_t=files[:3]

print(files_t)




for i in range(len(files_t)):
    print('new file')
    vidcap=cv2.VideoCapture(mypath+files_t[i])
    succ,images=vidcap.read()
    count=0
    while succ:
        cv2.imwrite('./dataset_2/training/images/data/'+str(count)+'.png',images)
        succ,images=vidcap.read()
        succ,images=vidcap.read()
        print(count)
        count+=1
