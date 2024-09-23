# https://www.youtube.com/watch?v=oXlwWbU8l2o
# For training
resourcesBasePath = r"C:\Users\Tanner\Documents\OpenCV"
facesPath = fr"{resourcesBasePath}\Resources\Faces"
photosPath = fr"{resourcesBasePath}\Resources\Photos"
videosPath = fr"{resourcesBasePath}\Resources\Videos"

tutorialFaces = fr"{resourcesBasePath}\TutorialFaces"
tutorialVideos = fr"{resourcesBasePath}\TutorialVideos"

#All imports 
import cv2 as cv 
import sys
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import os
"""Reading images"""
# img = cv.imread(fr"{tutorialFaces}\Cat.jpg")
# cv.imshow("Cat", img)
# cv.waitKey(0)

"""Reading videos"""
# capture = cv.VideoCapture(fr"{tutorialVideos}\DogVideo.mp4")
# while True:
#     isTrue, frame = capture.read() #Read video frame by frame, and bool saying if it was successful
#     cv.imshow('Video', frame)
#     if cv.waitKey(20) & 0xFF==ord('d'): #If the letter d is pressed, break
#         break
# capture.release()
# cv.destroyAllWindows()

"""Resize and rescale video frames
Good practice to rescale, to fit camera
"""
# def rescaleFrame(frame, scale=0.75):
#     #Images, existing videos
#     width = int(frame.shape[1] * scale)
#     height = int(frame.shape[0] * scale)
#     dimensions = (width, height)
#     return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# def changeRes(width, height):
#     #Live video only
#     capture.set(3, width) #3 is the width, 4 is the height
#     capture.set(4, height)

# capture = cv.VideoCapture(fr"{tutorialVideos}\DogVideo.mp4")
# while True:
#     isTrue, frame = capture.read() #Read video frame by frame, and bool saying if it was successful
#     frame_resized = rescaleFrame(frame)
#     cv.imshow('Video', frame)
#     cv.imshow('Video Resized', frame_resized)
#     if cv.waitKey(20) & 0xFF==ord('d'): #If the letter d is pressed, break
#         break
# capture.release()
# cv.destroyAllWindows()


"""Drawing shapes and putting text on images"""
# img = cv.imread(fr"{tutorialFaces}\Cat.jpg")

# blank = np.zeros((500, 500, 3), dtype='uint8') #Black image, uint8 is datatype of img
# # blank[:] = 0,255,0 #Green
# # blank[200:300, 300:400] = 0,255,0 #Green
# # (0,0) is top left, (250,250) is bottom right, (0,255,0) is color
# # cv.rectangle(blank, (0,0), (250,250), (0,255,0), thickness=cv.FILLED) #Filled rectangle
# cv.rectangle(blank, (0,0), (blank.shape[1]//2,blank.shape[0]//2), (0,255,0), thickness=cv.FILLED) #Filled rectangle
# # (250, 250) is center, 40 is radius, (0,0,255) is color
# cv.circle(blank, (250,250), 40, (0,0,255), thickness=-1) #Filled circle
# # Draws a point from (0,0) to (250, 250), with color (0,255,0)
# cv.line(blank, (0,0), (blank.shape[1]//2,blank.shape[0]//2), (255,255,255), thickness=3)
# # (255, 255) is top left, 1.0 is font scale, (0,255,255) is color, 2 is thickness
# cv.putText(blank, "Hello", (255,255), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0,255,255), 2)
# cv.imshow("Green", blank)

# cv.waitKey(0)


"""5 Essential functions in OpenCV 31:56"""

#1. Convert img to grayscale
# img = cv.imread(fr"{tutorialFaces}\Cat.jpg")
# cv.imshow("Cat", img)
# # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
# # cv.imshow("Gray", gray)

# #2. Blur image
# blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT) # img, colorSize to compute blur (has to be odd)
# # cv.imshow("Blur", blur)

# #3. Edge Cascade, find edges present in image
# canny = cv.Canny(img, 125, 175) # img, threshold1, threshold2. We can reduce edges by passing in the blur image instead of img
# # cv.imshow("Canny Edges", canny)

# #4. Dilating the image, thickening the edges
# dilated = cv.dilate(canny, (7,7), iterations=3) # img, kernel, number of iterations
# # cv.imshow("Dilated", dilated)

# #5. Eroding, opposite of dilating, gets back to edges image
# eroded = cv.erode(dilated, (7,7), iterations=1) # img, kernel, number of iterations
# # cv.imshow("Eroded", eroded)

# # 6. Resize
# resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC) # img, dimensions, interpolation
# cv.imshow("Resized", resized)
# #cv.INTER_CUBIC for shrinking
# #cv.INTER_LINEAR for making larger, cv.INTER_CUBIC is slower but higher quality

# #7. Cropping
# cropped = img[50:200, 200:400] # y1:y2, x1:x2
# cv.imshow("Cropped", cropped)

# cv.waitKey(0)


"""Image transformations (translation, rotation, resizing, flipping)"""
# img = cv.imread(fr"{tutorialFaces}\Cat.jpg")

# #Translation, shift along X, Y, or both
# def translate(img, x, y):
#     transMat = np.float32([[1,0,x], [0,1,y]]) #Translation matrix
#     dimensions = (img.shape[1], img.shape[0]) #Wiedth, height
#     return cv.warpAffine(img, transMat, dimensions)

# # -x --> Left
# # -y --> Up
# # x --> Right
# # y --> Down

# translated = translate(img, 100, 100)
# cv.imshow("Translated", translated)


# #Rotation, rotate by some angle
# def rotate(img, angle, rotPoint=None):
#     (height, width) = img.shape[:2] #height and width (first two values)
#     if rotPoint is None:
#         rotPoint = (width//2, height//2)
#     rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0) #Rotation matrix, rotPoint (center), angle, and scale (no scaling)
#     dimensions = (width, height)
#     return cv.warpAffine(img, rotMat, dimensions)

# rotated = rotate(img, 45)
# # cv.imshow("Rotated", rotated)

# #Resize image
# resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)
# # cv.imshow("Resized", resized)

# #Flipping
# flip = cv.flip(img, -1) #0 flips vertically, 1 flips horizontally, -1 flips both
# # cv.imshow("Flipped", flip)

# #Cropping
# cropped = img[200:400, 300:400]
# cv.imshow("Cropped", cropped)

# cv.waitKey(0)

"""Contour detection"""
# Contours: Outline
# Edges: Point where surfaces meet.
# How contours differ from edges: Continous vs line segments
# Guideline, use canny method first to find contours, then try the threshold method. The result image should be close to the original image

# img = cv.imread(fr"{tutorialFaces}\Cat.jpg")
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT) #Blur the image
# canny = cv.Canny(blur, 125, 175) #Grab edges


# #threshold is a binary image, anything above 125 is white, below is black
# ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY) #Thresholding, anything above 125 is white, below is black 
# # One way to find contours. Had to blur due to it being 1.5k contours, that's too high
# # contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE) #Find contours, cv.RETR_LIST retrieves all contours, cv.CHAIN_APPROX_NONE retrieves all points
# contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE) #Find contours, cv.RETR_LIST retrieves all contours, cv.CHAIN_APPROX_NONE retrieves all points


# blank = np.zeros(img.shape, dtype='uint8') #Create blank image
# # cv.imshow("Blank", blank)
# cv.drawContours(blank, contours, -1, (0,0,255), 1) #Draw contours on blank image, -1 is all contours, (0,0,255) is color, 1 is thickness
# cv.imshow("Contours Drawn", blank)
# # print(len(contours))

# cv.waitKey(0)

"""Color Spaces"""
# BGR, grayscale, etc
# img = cv.imread(fr"{tutorialFaces}\Cat.jpg")

# # BGR to grayscale
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# # cv.imshow("Gray", gray)

# # BGR to HSV
# hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# # cv.imshow("HSV", hsv)

# # BGR to L*a*b
# lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
# # cv.imshow("LAB", lab)

# # BGR to RGB
# rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# # cv.imshow("RGB", rgb)
# # plt.imshow(rgb) #Plt shows RGB, cv shows BGR
# # plt.show()

# #HSV to BGR
# hsv_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
# # cv.imshow("HSV to BGR", hsv_bgr)

# #LAB to BGR
# lab_bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
# # cv.imshow("LAB to BGR", lab_bgr)



"""Color channels"""
# Color image has RGB (color channels merged together), could make it red, green, blue

img = cv.imread(fr"{tutorialFaces}\Cat.jpg")
# b,g,r = cv.split(img) #Split the image into blue, green, red, shown in grayscale
# cv.imshow("Blue", b)
# cv.imshow("Green", g)
# cv.imshow("Red", r)

# print(img.shape) #3 at end is color channels
# print(b.shape)
# print(g.shape)
# print(r.shape)

# merged = cv.merge([b,g,r]) #Merge the color channels back together
# cv.imshow("Merged", merged)

# # Show color channels in their normal color
# blank = np.zeros(img.shape[:2], dtype='uint8') #Create blank image
# blue = cv.merge([b, blank, blank])
# green = cv.merge([blank, g, blank])
# red = cv.merge([blank, blank, r])

# cv.imshow("Blue", blue)
# cv.imshow("Green", green)
# cv.imshow("Red", red)

"""Blurring & smoothing"""
# Smooth when it has noise 
# Average blur: When you apply blur, you need to define the window. Computes the blur based on the pixel blurs at the center of the window, then it moves the window
# img = cv.imread(fr"{tutorialFaces}\Cat.jpg")
# cv.imshow("Cat", img)

# # Averaging
# average = cv.blur(img, (7,7)) #img, kernel size
# cv.imshow("Average Blur", average)

# # Gaussian blur
# # Gaussian: Same thing as average, but the surrounding pixels are given a weight, then this is used instead
# gauss = cv.GaussianBlur(img, (7,7), 0) #img, kernel size, sigmaX (standard deviation in x direction)
# cv.imshow("Gaussian Blur", gauss)

# # Median blur
# # Median: Looks at the pixels in the window, and takes the median value of the pixels in the window
# # More effective at reducing noise
# median = cv.medianBlur(img, 7) #img, kernel size (has to be odd)
# cv.imshow("Median Blur", median)

# # Bilateral blurring
# # Most effective: This blurs the image, but keeps the edges sharp
# bilateral = cv.bilateralFilter(img, 10, 35, 25) #img, diameter, sigmaColor (larger value, more colors considered), sigmaSpace (larger values, more pixels influenced by the center pixel)
# cv.imshow("Bilateral", bilateral)


"""Bitwise operations"""
# AND, OR, XOR, NOT



# blank = np.zeros((400,400), dtype='uint8') #Create blank image
# #starting point, ending point, color, thickness
# rectangle = cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1) #Create rectangle
# # center, radius, color, thickness
# circle = cv.circle(blank.copy(), (200,200), 200, 255, -1) #Create circle

# cv.imshow("Rectangle", rectangle)
# cv.imshow("Circle", circle)

# # Bitwise AND
# bitwise_and = cv.bitwise_and(rectangle, circle) #Intersect
# cv.imshow("Bitwise AND", bitwise_and)

# # Bitwise OR
# bitwise_or = cv.bitwise_or(rectangle, circle) #Union
# cv.imshow("Bitwise OR", bitwise_or)

# # Bitwise XOR
# bitwise_xor = cv.bitwise_xor(rectangle, circle) #Opposite of intersect
# cv.imshow("Bitwise XOR", bitwise_xor)

# # Bitwise NOT
# bitwise_not = cv.bitwise_not(rectangle) #Opposite of rectangle
# cv.imshow("Bitwise NOT", bitwise_not)


"""Masking"""
# Allows us to focus on parts of img we want to focus on (just the faces)
# img = cv.imread(fr"{tutorialFaces}\Cat.jpg")
# blank = np.zeros(img.shape[:2], dtype='uint8') #Create blank image, has to be same size
# # center, radius, color, thickness
# circle = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1) #Create circle mask
# # starting point, ending point, color, thickness
# rectangle = cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1) #Create rectangle mask
# mask = cv.bitwise_and(circle, rectangle) #Combine masks
# cv.imshow("Mask", mask)
# masked = cv.bitwise_and(img, img, mask=mask) #Apply mask to image
# cv.imshow("Masked Image", masked)

"""Computing Histograms"""

# img = cv.imread(fr"{tutorialFaces}\Cat.jpg")
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# blank = np.zeros(img.shape[:2], dtype='uint8') #Create blank image
# # center, radius, color, thickness
# circle = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1) #Create circle mask
# mask = cv.bitwise_and(gray, gray, mask=circle) #Apply mask to image
# cv.imshow("Mask", mask)

# #Grayscale histogram
# gray_hist = cv.calcHist([gray], [0], mask, [256], [0,256]) #Image, channels, mask, histSize (number of bins), ranges of pixel values
# plt.figure()
# plt.title("Grayscale Histogram")
# plt.xlabel("Bins")
# plt.ylabel("# of pixels")
# plt.plot(gray_hist)
# plt.xlim([0,256])
# plt.show()


"""Computing Color Histograms"""

# img = cv.imread(fr"{tutorialFaces}\Cat.jpg")
# # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# blank = np.zeros(img.shape[:2], dtype='uint8') #Create blank image
# # center, radius, color, thickness
# mask = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1) #Create circle mask
# masked = cv.bitwise_and(img, img, mask=mask) #Apply mask to image
# cv.imshow("Mask", masked)

# plt.figure()
# plt.title("Colour Histogram")
# plt.xlabel("Bins")
# plt.ylabel("# of pixels")

# #Color histogram
# colors = ('b', 'g', 'r')
# for i, col in enumerate(colors):
#     # img, channels, mask, histSize, ranges
#     hist = cv.calcHist([img], [i], mask, [256], [0,256])
#     plt.plot(hist, color=col)
#     plt.xlim([0,256])

# plt.show()

"""Thresholding"""
# # Simple thresholding: If pixel value is greater than threshold, it's assigned one value, else another value
# # Adaptive Thresholding: Allows computer to specify the value of threshold, based on the region of the image
# img = cv.imread(fr"{tutorialFaces}\Cat.jpg")
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow("Gray", gray)

# # Simple thresholding
# threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY) #Image, threshold (0 if below, 255 if above), max value, type
# cv.imshow("Simple Thresholded", thresh)

# #Inverse thresholding
# threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV) #Image, threshold (0 if below, 255 if above), max value, type
# cv.imshow("Simple Thresholded Inverse", thresh)

# # Adaptive thresholding
# adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3) #Image, max value, adaptive method, type, block size, constant
# cv.imshow("Adaptive Thresholding", adaptive_thresh)
# # cv.ADAPTIVE_THRESH_GAUSSIAN_C --> Gaussian weighted sum of the neighborhood, could work, could not, play around

"""Edge Detection"""

# img = cv.imread(fr"{tutorialFaces}\Cat.jpg")
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# # Laplacian
# lap = cv.Laplacian(gray, cv.CV_64F) #Image, data type
# lap = np.uint8(np.absolute(lap)) #Convert to unsigned int
# cv.imshow("Laplacian", lap)

# # Sobel
# # Sobel: Computes the gradient of the image, in x and y direction
# sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0) #Image, data type, x direction, y direction
# sobely = cv.Sobel(gray, cv.CV_64F, 0, 1) #Image, data type, x direction, y direction
# # cv.imshow("Sobel X", sobelx)
# # cv.imshow("Sobel Y", sobely)

# combined_sobel = cv.bitwise_or(sobelx, sobely) #Combine the two
# cv.imshow("Combined Sobel", combined_sobel)

# canny = cv.Canny(gray, 150, 175) #Image, threshold1, threshold2
# cv.imshow("Canny", canny)

# # Canny is mostly used for clean edges
# # In more advanced cases, sobel will be used to find the edges, then canny will be used to clean them up
# # Laplacian isn't used often

# cv.waitKey(0)


"""Face detection --> pre-built haar cascades"""

# haarcascades: Pretrained classifiers for detecting objects, not recognition
# Better, more complex version of haarcascades: Local Binary Patterns Histograms (LBPH)
# Called Dlib’s HOG + Linear SVM: This method uses Histogram of Oriented Gradients (HOG
# haar_faces_path = fr"{resourcesBasePath}\Section #3 - Faces\haar_face.xml"
# face = r"C:\Users\Tanner\Documents\OpenCV\TutorialFaces\Face.jpg"
# # print(haar_faces_path)
# img = cv.imread(face)
# # cv.imshow("Person", img)

# # Face detection does not use color
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# # Read haarcascade
# haar_cascade = cv.CascadeClassifier(haar_faces_path)

# # Detect faces
# # Doesn't work well with small images
# faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3) #Image, scale factor, min neighbors
# # Can increase minNeighbors to reduce false positives
# # scaleFactor: How much the image size is reduced at each image scale

# print(f'Number of faces found = {len(faces_rect)}')

# for (x,y,w,h) in faces_rect:
#     # Draw rectangle around face
#     # (x,y) is top left, (x+w, y+h) is bottom right, (0,255,0) is color, thickness is 2
#     cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)

# cv.imshow("Detected Faces", img)
# cv.waitKey(0)


"""Face recognition --> pre-built haar cascades"""
facesPath = r"C:\Users\Tanner\Documents\OpenCV\Resources\Faces\train"
people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
haar_faces_path = fr"{resourcesBasePath}\Section #3 - Faces\haar_face.xml"

haar_cascade = cv.CascadeClassifier(haar_faces_path)
# features = []
# labels = []


# def create_train():
#     for person in people:
#         path = os.path.join(facesPath, person)
#         label = people.index(person) #Index instead of string for mapping back

#         for img in os.listdir(path):
#             img_path = os.path.join(path, img)
#             img_array = cv.imread(img_path) #read img
#             gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
#             faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

#             for (x,y,w,h) in faces_rect:
#                 faces_roi = gray[y:y+h, x:x+w]
#                 features.append(faces_roi)
#                 labels.append(label)

# create_train()
# print(f'Length of features = {len(features)}')
# print(f'Length of labels = {len(labels)}')

# features = np.array(features, dtype='object')
# labels = np.array(labels)

# # # Train the recognizer
# face_recognizer = cv.face.LBPHFaceRecognizer_create() #Built model
# face_recognizer.train(features, labels)
# face_recognizer.save(fr"{resourcesBasePath}\TutorialFaces\face_trained.yml")
# np.save(fr"{resourcesBasePath}\TutorialFaces\features.npy", features)
# np.save(fr"{resourcesBasePath}\TutorialFaces\labels.npy", labels)




# Within new file
features = np.load(fr"{resourcesBasePath}\TutorialFaces\features.npy", allow_pickle=True)
labels = np.load(fr"{resourcesBasePath}\TutorialFaces\labels.npy")
face_recognizer = cv.face.LBPHFaceRecognizer_create() #Built model
face_recognizer.read(fr"{resourcesBasePath}\TutorialFaces\face_trained.yml")

img = cv.imread(fr"{resourcesBasePath}\Resources\Faces\val\elton_john\1.jpg")
# print(img)
# print(fr"{resourcesBasePath}\Resources\Faces\val\elton_john\1.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# C:\Users\Tanner\Documents\OpenCV\Resources\Faces\val\1.jpg
# C:\Users\Tanner\Documents\OpenCV\Resources\Faces\val\
# cv.imshow("Person", gray)

#Detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')

    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)

cv.imshow("Detected Face", img)
cv.waitKey(0)