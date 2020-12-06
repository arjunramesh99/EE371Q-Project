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
from keras.models import load_model

TEST = True
REMOVE_NEUTRAL = False
TEST_PREFIX = "test_face"

emotion = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']

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



class VideoStreamModel:

    def __init__(self, model_path):
        self.__ct = 0;
        self._model = load_model(model_path)
        self._face_obj = FaceImageProcessor()
    
    def display_top3_emoji(self, top1, top2, top3):
        base_path = Path('./emotion_images')
        top1_img = Image.open(base_path / str(top1))
        top2_img = Image.open(base_path / str(top2))
        top3_img = Image.open(base_path / str(top3))
        plt.subplot(131)
        plt.imshow(top1_img)
        plt.title(f"Top 1: {emotion[top1]}")
        plt.axis('off')

        plt.subplot(132)
        plt.imshow(top2_img)
        plt.title(f"Top 2: {emotion[top2]}")
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(top3_img)
        plt.title(f"Top 3: {emotion[top3]}")
        plt.axis('off')

        plt.show()


    def start_webcam(self):
        self._vid_stream = cv2.VideoCapture(0)
        if (self._vid_stream.isOpened()):
            print("Webcam successfully accessed!\n")
        
    def stream_capture_predict(self):
        '''
            Continuous video stream.
            Press 'q' to quit, and 'c' to capture
        '''
        while self._vid_stream.isOpened():
            captureKey = cv2.waitKey(10) & 0xFF
            if (captureKey == ord('q')):
                break
            ret, frame = self._vid_stream.read()
            cv2.imshow('frame', frame)        
        
            # Capture and Predict
            if captureKey == ord('c'):
                pil_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_face_img = Image.fromarray(pil_frame)
                print("Image captured!")
            
                # Perform face detection on captured image
                self._face_obj.set_base_img( np.array(pil_frame) )
                processed_face = self._face_obj.process_face_from_img( resize=(100,100) )
                if processed_face:
                    print("Face Detected!\n")
                    # Debugging
                    if TEST:
                        self._face_obj.show_process_result()
                        processed_face.save(f"{TEST_PREFIX}_{self.__ct}.jpg")
                        self.__ct = self.__ct + 1;

                    # Predict emotion using ML model
                    face_np = np.array(processed_face)
                    face_np_arr = np.expand_dims(face_np, axis=0) 
                    predictions = self._model.predict(face_np_arr, verbose=1)
                    if REMOVE_NEUTRAL:
                        predictions = np.delete(predictions, 6, 1)
                    print(f"Preds: {predictions}")
                    top_preds = np.argsort(predictions)[0]
                    print(f"Top Preds: {top_preds}")
                    self.display_top3_emoji(top_preds[-1], top_preds[-2], top_preds[-3])

                else:
                    print("Unable to find face. Please look straight at the camera\n")

                
                

    def __del__(self):
        print("destructor")
        if self._vid_stream.isOpened():
            self._vid_stream.release()
        cv2.destroyAllWindows()
        
        if TEST:
            print("Destructor")
            file_list = list(Path('.').glob(f'{TEST_PREFIX}*'))
            for f in file_list:
                try:
                    os.remove(f)
                except:
                    print(f"File {f} doesn't exist")




class App:

    def __init__(self, model_path):
        self._video = VideoStreamModel(model_path)
                    
    def deploy(self):
        self._video.start_webcam()
        self._video.stream_capture_predict()


if __name__ == '__main__':
    app = App("./models/best_model_2048_frz_25_test_41.h5")
    app.deploy()
    del app
