# Before evaluating our model and connecting it through our code, we will need create a virtual environment 
# where we install the Python packages necessary for the specific application that we will be  building. To 
# begin we will create a new environment with: 
#   > conda create --name myenviron 	
# Now to activate the environment for windows you would simply write: 
#   > activate myenviron
# If you are using the Linux or Mac:
# 	> source activate myenviron
# Now run our model and connect it to the Udacity Simulator in order to drive the car. To do so we need to 
# write a script to set up a 'bi-directional client server communication'. This code we re about to write is not 
# related to deep learning, but it is a step we need to take to run the model on our car.
# How the connection is created between the model and our simulator?
# in this file we are setup the connection betwwen our model and the simulator in order to get the car to drive
# based on the steering angles provided by the trained model. To setup this connection we must install 'Python Socket
# IO server' but before we event get to that we will need to initialize a 'Python web application' and to do so we must
# install 'flask', so we run:
#   > conda install -c anaconds flask
# Once your installation is complete back to our editor (here) we will import from 'flask' import 'Flask' class data type
# which is what we are going to use to create instances of a web appliaction.
# FLask is a python micro framework that is used to build we apps, we can initialize our application:
#   app = Flask(__name__) #'__main__'
# Thereby creating an instance of the flask class for our webapp for now this will take a special veraible known as the name
# which will end up having the value of __main__ like so. 
# For example When executed above code, what ww will do is define a function:
#
# @app.route('/home')
# def greeting():
#   return "welcome!"

# what we can do is specify a router decorator defined by  '@app.route('/home'). This rout decorator we use , it to tell flask 
# what url we should use to trigger our function. It is going to be a simple local host url with path home the decorator 
# registers this function for this given url rule. If user navigates to this path on the browser then the function will run
# and return the output to our clients. 
# Now what is the significance of the __name__, well whenever you execute a Python script, Python assigns the name __main__,
# So as we are executing the script __name__ will equal the string value __main__.
# So what we can do is check:
#
# if __name__ == '__main__': 
#   app.run(port=3000)
#
# so when the app is run what we want to do is listen on port=3000, but what we are doing is ensuring that the app listens to
# any connections that occur on this port localhost:3000, so running above code on our terminal:
#   > python drive.py
# Then go to browser on url type: localhost:3000/home we will see the welcome! message. Thus we have a web app with some 
# content returned by python.
# 
# Ok but this is not what we are actually aiming for but this was a good intro to flask. The goal is to create some kind of
# bi-directional client server communication like what we just saw. But ultimately as an end result create a connection 
# between our model which we will load into our code and the simulator. This server will be initialized with 'Socket.IO'  
# First to install the socket server with conda:
#   > conda install -c conda-forge python-socketio
# after install, we will mport socketio. and now we need initialize our web server where we are going to set 'sio' for socketio to 
# 'socketio.Server()'
# web sockets in general are used to perform real time communication between a client and server when a client creates a single
# connection to a web socket server, it keeps listening for new events from the server allowing us to continuously update the
# client with data. In our case we will be setting up a socket that IO server and establish bi-directional communication with the
# simulator this server class implements a fully compliant socket io web server and now having specified the server we will requirea
# middleware what is called  middleware to dispatch traffic to a socket.io web application. Simply stated we will combine our socket 
# server with a flask webapp.
# So what we do is sets app  to 'app = socketio.Middleware(sio, app)'
# We can make use of a webserver for gateway interface WSGI to have our web server send any requests made by the client to the web 
# appliaction itself. To launch this WSGI server, we simply create a socket and having already done so we then call: 
#   eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
# and we have install eventlet:
#   > conda install -c conda-forge eventlet
# 
# Now having opened up a listening socket establishing the connection what we want to do is when there is a connection with the client 
# we want to fire off an event handler. To register an event handler we use @sio.on('connect'), we make use of connect which fires upon
# a connection and will invoke the following function: 
#   @sio.on('connect') 
#   def connect(sid, environ):
#       print('Connected')
#
# Now what we can dois actually go ahead and run the simulation on autonomous mode. First we will go to terminal and go to our project
# thenmake sure activate our environment:
#   > activate myenviron
# and now run the drive.py
#   > python drive.py
# and is shoudld everything works... then go to similator run it on the autonomous mode. then we should see in our terminal the line says
# connected. we have effectively established a connection.
# now before we obtain steering and throttle values based on the model, lets just get the car to start off by driving straight as soon as
# the connection is made so we are going to define a new function called 'send_control()' which is going to take in the steering angle
# and throttle value, and as soon as connects we will invoke this function send control and at first we want to drive straight so the 
# steering value will be zero, but at the same time we will give it a throttle power of 1. and we want to emit this data to our simliator
# and we do this by socketio.emit() and we are emitting with custom event name it 'steer' and it is going to listen for data that we send 
# it and we will send this data in the form of key value pairs. For the key 'steering_angle' we send it the steering that we passed in, but 
# we will enmit it as a string value ( steering_angle.__str__() ) that is how it will be processed to ultimately make the udacity simulator 
# work. and we do the same thing for throttle and emit the throttle value that is being passed it as string ( throttle.__str__() )

# when we run the drive.py again, and load the similator in autanomus mode you should see the car driving straight. 
# so what we want to do is load our model and actually have it give the car appropriate steering angles based on which part of the track it is 
# in.

# However you must make sure that the car is initially stationary so we will set the throttle to zero to start with and the steering angles 
# themselves will be determined based on the models predications. So we must load the model that we saved earlier to do so we must first load
# our model from 'Keras.models' we will import 'load_model', but before proceeding we must install both tensorflow and keras into our environment
# go to your terminal and install them:
#   > conda install -c conda-forge tensorflow
#   > conda install -c conda-forge keras

# so after we use these to load our model: model = load_model('model.h5')
# before making use ofthis model we need to register a specific event handler, once again we write '@sio.on('telemetry')', which will fire a 
# function def telemetry(sid, data) sid = sessionId, data = data  received from the similator. So make sure that you get the naming right and 
# in the case of an event the function is fired with the aapropriate data. Essentially what is going to happen is as soon as the connection is 
# established we are setting the initial steering andthrottle values and emitting it to the simulator such that the car starts off as stationary
# andfacing forward, but then the simulator will send us back data which contains the current image of the frame where the car is presently located 
# in the track and based on the image we want to run it through our model. The model will extract the features from the image and predict the steering
# angle which we send back to the simulation, and we keep doing that for the entire simulation such that the car starts driving on it own.
# so first we obtain the current image as data image which is base 64 encoded, we must decode it. So we are going to import base64 for this and use the
# base64.b64decode(image), and before we can open identify the given image file with 'image.open()' from python imaging library, we need to use a 
# buffer module to mimic our data like a normal file which we can further use for processing. to do so we make use BytesIO, which we will import it
# from io.  
# we have to install python image librery: > conda install -c anconda pillow
# allowing us to make use of python image library and recal as done previously we must convert our data to an array:
#   image = np.asarray(image)
# since this image is what is being fed into the NN to predict the appropriate steering angle, it must be preprocessed the same way as the images we use
# to train our model. So from colab we copy the code 'img_preprocess' to here. this will preprocess our image the same way we had it earlier. The we net 
# the new image value to 'img_preprocess', this preprocessing it accordingly. And now the model if you were to test it actually expects 4D arrays whereas
# our image is only 3D, so we enclose this image inside of another arrayby updating the image veriable to:
#   image = np.array([image])
# and now we can feed the image into the model that we previously loaded so that it predicts an appropriate steering angle for that image, ultimately
# helping the car steer based on the current location within the track.
# so we set the steering_angle to 'model.predict(image)' and make sure the output os the steering_angle ends up being a float and we will send that steering
# angle over to the simulation with send control steering angle and now for throttle, we will just assign it a throttle value of one point:
#   send_control(steering_angle, 1.0) 
#
# so overall in the ground scheme of things what is going to happen as soon as the simulator connects we are setting the initial steering and throttle 
# values to the simulator and sending it to the simulator so that the car starts off stationary and facing forward, but then the simulator will send us 
# data, it is going to send our model images illustrating the car's current location which is processed in our model.predict function predicting a 
# steering angle for every location of the simulation that is being sent to it sending the simulation back the appropraite steering angles helping it
# drive accordingly.

# we run the code and simulator, but the car speeding up, so the first thing to note is that we don't want it to reach maximum speed, rather we want to
# set a speed limit.
# so we are going to add some logic to ensure that the car always moves at a constant speed.
# what the simulator also does as it sends us the data about the car's current speed which is accessible to us as data speed we will make sure that
# it is a float value and store it inside of a speed variable:
#   speed = float(data['speed'])
# also we specify speed_limit as 10
# we can enforce the speed limit by setting the throttle value: throttle = 1.0 - (speed / speed_limit)
# how does this ensure that our car doesn't surpass the speed limit, well the initial speed is going to be zero, so zero divide by 10 is zero, 
# so the throttle will begin by being 1.0 the car will then keep speeding up but then realize that by the time the speed of the car reaches 10, then 
# then throttle = 0 and thus enforcing a constant speed at the speed limit.
# so what we will actually do is end up printing 3 placeholders for current steering_angle, throottle, and speed.
# after running the simulator again, we see that in the beginning do fine but eventually swaying left and right and finally gets out of the road.

# we noticed that it was swaying side to side a bit in that it seemed to have trouble navigating the road as soon as it approached the ater on the side 
# of the road this can be attributed to many factors, however our small data set is most likely a contributing factor. we can even run this model on our 
# second track and eveluate how it perfoms for the sake of experimentation.
# this means our model most likely has not learned to generalize to a new dataset



import socketio
import eventlet
import numpy as np
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2
    
sio = socketio.Server()
    
app = Flask(__name__) #'__main__'

speed_limit = 10

def img_preprocess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img

@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    throttle = 1.0 - (speed / speed_limit)
    print('{} {} {}'.format(steering_angle, throttle, speed))
    # send_control(steering_angle, 1.0)
    send_control(steering_angle, throttle)

     
@sio.on('connect') # reserved 3 names; connect, message, and disconnect
def connect(sid, environ):
    print('Connected')
    send_control(0, 1)


def send_control(steering_angle, throttle):
    sio.emit('steer', data = {
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })
    
if __name__ == '__main__':
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

