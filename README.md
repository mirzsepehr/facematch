
# Face Match 

This is a face match project written in python language. This project has written in **fastapi** framework using YuNet neural network for face detection and pretrained ResNet neural network for face recognition task. Generally the code detects faces in the picture, then extracts them, and finally recognizes them.  

## Running Program in venv

The file structure of the project depends on the way you want to run the program. If you want to run the program without docker and use the python interpreter, you should do the following:
- install anaconda on your desktop which is available on this link **[this link](https://docs.anaconda.com/anaconda/install/)**
- move the "__face_detection_yunet_2022mar.onnx__"  file to the app folder and rename this specific file to "__yunet.onnx__".  This is our face detection model.
- open cmd in the project's folder. In my case it is:
> C:\Users\Sepehr\Desktop\project2
- run the following command in order to create a virtual environment with the specific python version: 
> conda create --name myvirtualenv python==3.12
- Then activate your virtual environment:
> conda activate myvirtualenv
- install the requirements:
>pip install -r requirements.txt
- in order to run the program we need uvicorn which will be installed automatically with running this command:
> pip install fastapi[all]
- And finally get to the app folder and run the program using uvicorn:
> cd app
> uvicorn main:app

if you want to exert changes to the code and see the changes simultaneously add __ --reload__ to the end of your command. You can add host and port and workers as well! For instance:
> uvicorn main:app --host 0.0.0.0 --port 8000 --workers 5 --reload

You can test the facematch post method by visiting the __/docs__ path in your url path. For instance:
> http://0.0.0.0:8000/docs   

Visit **[this link](https://fastapi.tiangolo.com/)** in order to read and learn more about fastapi framework.


## Run Program with Docker

In order to run the program inside a docker container, just clone this project inside a folder:
> git clone https://github.com/mirzsepehr/facematch
> cd facematch

Then you can run the "__docker compose__".
> docker compose up --build

The webpage will be available in the 127.0.0.1 
Make sure that you have docker installed in your computer!