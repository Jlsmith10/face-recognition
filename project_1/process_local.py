"""
ECE196 Face Recognition Project
Author: Will Chen

Prerequisite: You need to install OpenCV before running this code
The code here is an example of what you can write to print out 'Hello World!'
Now modify this code to process a local image and do the following:
1. Read geisel.jpg
2. Convert color to gray scale
3. Resize to half of its original dimensions
4. Draw a box at the center the image with size 100x100
5. Save image with the name, "geisel-bw-rectangle.jpg" to the local directory
All the above steps should be in one function called process_image()
"""

# TODO: Import OpenCV
import cv2


# TODO: Edit this function
def process_image():
    # Read in geisel image
    geisel_img = cv2.imread("geisel.jpg")
    
    # Convert to grayscale
    gray_geisel_img = cv2.cvtColor(geisel_img, cv2.COLOR_BGR2GRAY)
    
    orig_size = gray_geisel_img.shape

    print("orig size: ", orig_size)

    half_geisel = cv2.resize(gray_geisel_img, (0,0), fx=0.5, fy=0.5)
    new_size = half_geisel.shape

    print("New size: ", new_size)


    center = (new_size[0] // 2, new_size[1] // 2)
    top_left = (center[0] - 50, center[1] - 50)
    bottom_right = (center[0] + 50, center[1] + 50)

    rect_geisel = cv2.rectangle(half_geisel, top_left, bottom_right, color=255,     thickness=5)

    cv2.imwrite("rect_geisel.jpg", rect_geisel)
    
    return

# Just prints 'Hello World! to screen.
def hello_world():
    print('Hello World!')
    return

# TODO: Call process_image function.
def main():
    process_image()
    return


if(__name__ == '__main__'):
    main()
