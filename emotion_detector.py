#!/usr/bin/python3
import face_recognition
import numpy as np
import os
import cv2
import glob
from PIL import Image
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from pathlib import Path


TEST = True
TEST_PREFIX = "test_face"
class FaceImageProcessor:
    
    def __init__(self, imgpath = None):
        if imgpath is not None:
            self._base_img = face_recognition.load_image_file(imgpath)

    def __process_img(self, face_location, resize_dim = None):
        top, right, bottom, left = face_location
        face_arr = self._base_img[top:bottom, left:right]
        self._face = Image.fromarray(face_arr)
        self._resize_scale = resize_dim
        self._processed_face = self._face.resize(resize_dim) if resize_dim is not None\
                                        else self._face
        return self._processed_face 


    def show_process_result(self):
        plt.subplot(131)
        plt.imshow(self._base_img)
        plt.title("Base Image")
        plt.axis('off')

        plt.subplot(132)
        plt.imshow(self._face)
        plt.title("Face Location")
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(self._processed_face)
        plt.title("Processed Face")
        plt.axis('off')

        plt.show()


    def get_base_img(self):
        return self._base_img

    def get_face_img(self):
        return self._face_img

    def get_face_coordinates(self):
        return self._face_locations[0]

    def set_base_img(self, img):
        self._base_img = img

    def process_face_from_img(self, resize = None):
        self._face_locations = face_recognition.face_locations(self._base_img)
        if self._face_locations:
            processed_face = self.__process_img(self._face_locations[0], resize)

        return processed_face if self._face_locations else False



class VideoStreamer:

    def __init__(self):
        self.__ct = 0;
        self._face_obj = FaceImageProcessor()
    
    def start_webcam(self):
        self._vid_stream = cv2.VideoCapture(0)
        if (self._vid_stream.isOpened()):
            print("Webcam successfully accessed!\n")
        
    def stream_capture_image(self):
        while self._vid_stream.isOpened():
            c = cv2.waitKey(10) & 0xFF
            if (c == ord('q')):
                break
            self.__capture_image(c == ord('c'))


    def __capture_image(self, capture = False):
        ret, frame = self._vid_stream.read()
        cv2.imshow('frame', frame)        
        
        if capture:
            pil_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_face_img = Image.fromarray(pil_frame)
            print("Image captured!")
            
            # Perform face detection on captured image
            self._face_obj.set_base_img( np.array(pil_frame) )
            processed_face = self._face_obj.process_face_from_img( resize=(100,100) )
            if processed_face:
                print("Face Detected!\n")
                self._face_obj.show_process_result()
            else:
                print("Unable to find face. Please look straight at the camera\n")

            # Predict emotion using ML model


            # Debugging
            if TEST:
                pil_face_img.save(f"{TEST_PREFIX}_{self.__ct}.jpg")
                self.__ct = self.__ct + 1;


    def __del__(self):
        if (self._vid_stream.isOpened()):
            self._vid_stream.release()
        cv2.destroyAllWindows()
        
        if (TEST):
            file_list = list(Path('.').glob(f'{TEST_PREFIX}*'))
            for f in file_list:
                try:
                    os.remove(f)
                except:
                    print(f"File {f} doesn't exist")


if __name__ == '__main__':
    Video = VideoStreamer()
    Video.start_webcam()    
    Video.stream_capture_image()

    #Face_obj = FaceImageProcessor(args.imagepath)
    #Face_obj.process_face_from_img(resize=(100,100))
    #Face_obj.show_process_result()
