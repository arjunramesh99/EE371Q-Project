# Face-To-Emoji Generator

## Dataset and Model
I use the [Real-World Affective Faces Database](http://www.whdeng.cn/raf/model1.html), used for non-commercial research purposes only. They have firm guidelines on redistribution of content, so please request access from them if you wish to gain access to the dataset.

Users can access and download my trained model at the following [link](https://drive.google.com/drive/folders/1O1qQXRtKiZi57ihtrIOaE8CuMVNcB7At?usp=sharing). The model can recognize between the 7 classes of basic emotions

To integrate the model successfully to the code, add it under a new directory named *models*

## Usage
After downloading the model, run the `emotion_detector.py` script to start the webcam. The following are the controls to use the application:
* 'c' to capture image and match appropriate emoji
* 'q' to quit the application

