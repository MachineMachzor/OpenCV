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
