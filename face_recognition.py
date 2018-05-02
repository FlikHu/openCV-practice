#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 18:46:06 2018

@author: Flik
"""
import cv2

#Haar cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

#Face detection
def detect(original, grayscale):
    
    #Detect faces
    faces = face_cascade.detectMultiScale(grayscale, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(original, (x, y), (x+w, y+h), (0, 255, 255), 2)
        roi_gray = grayscale[y:y+h, x:x+w]
        roi_color = original[y:y+h, x:x+w]
        
        #Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (x1, y1, w1, h1) in eyes:
            cv2.rectangle(roi_color, (x1, y1), (x1+w1, y1+h1), (255, 255, 0), 2)
            
    return original

#Enable built-in webcam
video =cv2.VideoCapture(0)

while True:
    _, frame = video.read()
    grayscale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    canvas = detect(frame, grayscale)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
