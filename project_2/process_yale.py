#Import the os module, for the os.walk function
import os
import math
import random


project_dir = "/home/pi/face-recognition-master"
# Set the directory you want to start from
rootDir = os.path.join(project_dir, "images")

def remove_large_angle():
    for dirName, subdirList, fileList in os.walk(rootDir):
        print('Found directory: %s' % dirName)
        for fname in fileList:
            if(fname[-4:] == ".jpg" and not "Ambient" in fname):
            
                #print(fname)
                #print(fname[-19:-17]) 
                azimuth = int(fname[-19:-17])
                if(azimuth >= 15):
                    #print("Removing ", fname)
                    fpath = os.path.join(rootDir, dirName, fname)
                    print("fpath: ", fpath)
                    os.remove(fpath)

def split_train_test_val():
    for dirName, subdirList, fileList in os.walk(rootDir):

        split_path = os.path.split(dirName)
        # Prevent recursive nonsense
        if(split_path[-1] == "train" or split_path[-1] == "test" or split_path[-1] == "val"):
            
            print("Directory name is: ", dirName, " continuing")
            continue

        # README case
        if(len(fileList) == 1):
            continue

        print('Found directory: %s' % dirName)
        num_images = len(fileList)
        train_amount = int(0.7*num_images)
        test_amount = int(math.ceil(0.2*num_images))
        val_amount = num_images - train_amount - test_amount

        print("train:", train_amount) 
        print("test:", test_amount) 
        print("val:", val_amount) 
        
        random_arrangement = random.sample(range(0, num_images), num_images)
        
        train_images = fileList[:train_amount]
        test_images = fileList[train_amount:train_amount+test_amount]
        val_images = fileList[test_amount:test_amount+val_amount]

        print(len(train_images))
        print(len(test_images))
        print(len(val_images))

        # Make directories for each subfolder
        """
        os.mkdir(os.path.join(dirName, "train"))
        os.mkdir(os.path.join(dirName, "test"))
        os.mkdir(os.path.join(dirName, "val"))
        """
        move_files_to_dir(train_images, dirName, "train")
        move_files_to_dir(test_images, dirName, "test")
        move_files_to_dir(val_images, dirName, "val")

def move_files_to_dir(files, original_path, dir_name):
    for filename in files:
        print("Filename: ", filename)
        orig_file_path = os.path.join(original_path,  filename)
        print("Original path: ", orig_file_path)
        new_file_path = os.path.join(original_path, dir_name, filename) 
        print("New path: ", new_file_path)

        os.rename(orig_file_path, new_file_path)



def main():
    #remove_large_angle()
    split_train_test_val()


if __name__ == "__main__":
    main()



