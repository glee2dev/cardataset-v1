import shutil
import random
import os
dirpath = r'data2\JPEGImages'
destDirectory = r'Car_Dataset\test'

filenames = random.sample(os.listdir(dirpath), 285)
print(filenames)
for fname in filenames:
    srcpath = os.path.join(dirpath, fname)
    shutil.move(srcpath, destDirectory)

print('done!')

# filenames = os.listdir(destDirectory)

# print(filenames)