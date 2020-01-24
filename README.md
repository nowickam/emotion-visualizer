# emotion-visualizer
The program is a realization of the main assignment for ID1214 Artificial Intelligence course at Kungliga Tekniska Högskolan (Stockholm), done in collaboration with David Rundel. 

<b>It is a a generative art-based animation portraying the user’s mood.</b> We use two artificial-intelligence-based modules in the project - face detection (OpenCV) and emotion recognition (Convolutional Neural Network). These components serve as means for obtaining data (input in a form of face image from webcam) and processing it (evaluating the emotion). The output of the program is a generative-art-based function which takes as a parameter the determined emotion, translates it into a set of variables (color, brush weight etc.) and on this basis produces an image.

The resulting program is a simple application with two screens as a visual output, number of modules facilitating a real-time flow of data and artificial-intelligence-based logic and manipulation of the data.

When establishing the environment for the program, we recommend using Anaconda. In addition, a webcam is needed to capture images of the user. The expected output, when launching the program (integrated_system.py), is a full-screen application with start button triggering an emotion-driven animation. One quits the application with ESC button.

Examplary output:
![emotion_4](https://user-images.githubusercontent.com/49707233/73084242-e71e6200-3ecc-11ea-8b12-786740fd45ce.png)

Libraries required for the Integrated System (integrated_system.py):
- tensorflow 2.0.0
- keras 2.3.1
- numpy 1.17.3
- tkinter 8.6.8
- PIL 6.2.1
- opencv-python 4.1.2.30
- cairo 1.14.12
- pycairo 1.18.2

Necessary files for the Integrated System:
- model.h5
- model.json
- haarcascade_frontalface_default.xml

Libraries required for the Facial_Expression (facial_expression.ipynb):
- tensorflow 2.0.0
- keras 2.3.1
- numpy 1.17.3
- pandas 0.25.3
- hyperopt 0.1.2
- scipy 1.3.1
- sklearn 0.22
- matplotlib 3.1.1
