# Import the os module, for the os.walk function
import os
import math
import random
from scipy import misc
from matplotlib import pyplot as plt
import cv2

print(cv2.__version__)


# On RPI
# project_dir = "/home/pi/face-recognition-master"

# On James' Comp
project_dir = "/home/james/UCSD/ECE_196/face-recognition"

# Set the directory you want to start from
rootDir = os.path.join(project_dir, "images")


def remove_large_angle():
    for dirName, subdirList, fileList in os.walk(rootDir):
        print('Found directory: %s' % dirName)
        for fname in fileList:
            if(fname[-4:] == ".jpg" and not "Ambient" in fname):

                # print(fname)
                # print(fname[-19:-17])
                azimuth = int(fname[-19:-17])
                if(azimuth >= 15):
                    # print("Removing ", fname)
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


def reorganize_data():
    for dirName, subdirList, fileList in os.walk(rootDir):
        split_path = os.path.split(dirName)
        full_path = split_path[0].split("/") + [split_path[1]]
        if(len(full_path) == 9):
            curr_class = full_path[-2]
            curr_dir = full_path[-1]
            print("Class: ", curr_class, " Dir: ", curr_dir)
            print(full_path)

            # Take the contents and move them into
            new_dir_path = os.path.join(
                rootDir, project_dir, "new_images", curr_dir, curr_class)

            try:
                os.mkdir(new_dir_path)
            except:
                print("Hey")

            for file in fileList:
                src_filename = os.path.join(dirName, file)
                print("Source: ", src_filename)
                dest_filename = os.path.join(new_dir_path, file)
                print("Dest: ", dest_filename)

                os.rename(src_filename, dest_filename)


def resize_our_images():
    face_cascade = cv2.CascadeClassifier(
        '/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml')
    for dirName, subdirList, fileList in os.walk("../data"):
        split_path = os.path.split(dirName)
        full_path = split_path[0].split("/") + [split_path[1]]

        if(full_path[-1] == '17' or full_path[-1] == '18'):
            print(full_path)

            print(fileList)
            for file in fileList:
                image = cv2.imread(os.path.join(dirName, file))

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(gray, (x, y), (x+max(w, h), y+max(w, h)),
                                  (255, 255, 255), 2)

                    new_image = image[y:y+max(w, h), x:x+max(w, h)]
                    new_image_gray = cv2.cvtColor(
                        new_image, cv2.COLOR_BGR2GRAY)
                    resized_gray = cv2.resize(
                        new_image_gray, dsize=(224, 224))

                cv2.imshow("Frame", resized_gray)
                cv2.imwrite(os.path.join(dirName, "new_" + file), resized_gray)

                key = cv2.waitKey(1) & 0xFF


def main():
    # remove_large_angle()
    # split_train_test_val()
    # reorganize_data()
    resize_our_images()


if __name__ == "__main__":
    main()
