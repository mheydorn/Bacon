from __future__ import absolute_import
from __future__ import division

import threading
import Queue
import os
import sys
import csv
import json
import argparse
from IPython import embed
import numpy as np
import cv2
import time
import glob
import scipy.misc as scipy
from PIL import Image
import numpy as np
import tensorflow as tf
import math
from model import CNNModel
import pickle
import math
import gtk
import matplotlib.pyplot as plt
from numpy import zeros, newaxis

#Disable Cuda
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

#User adjustable parameters
debug = True
blurrImage = False
useLED = False #Turn LED on when car found (Odroid only)
numCNNThreads =  1 #Thraeds to use for CNN
readFromFile = True #Wither to read images from file (False = read from webcam)
writeToFile = False #Wither to write images to file 
#InputImagesFileLocation = '/media/ecestudent/43CBC47963CEF219/DataSets/Train/video10/raw/*.jpg' #If readFromFile is True
#InputImagesFileLocation = '/media/ecestudent/43CBC47963CEF219/DataSets/Test/video7/*.jpg'
#InputImagesFileLocation = '/media/ecestudent/43CBC47963CEF219/DataSets/ActionCam/CascadePositives/*.jpg'
InputImagesFileLocation = '/media/ecestudent/43CBC47963CEF219/DataSets/Negatives/YoutubeNegatives/RandomelyCroppedNegatives/*'
showFrame = True #Show frames with cv2.imshow as they are proccessed
useMultiframeFiltering = False

#Filter parameters:
thresh = 75 #How close a box must be to another box to be considered the same box
expirationTime = 1   #Number of frames a box must be idle (no matches found) before being removed
minCountsToShow = 1  #Number of validations a box needs before it will be shown
CNNSaidNoThreshold = 4000000 #Number of times the CNN can say no to a box in a row before the box is deleted


#Semaphores
ready = False
getNewFrame = True
gotNewFrame = False

if useLED:
  os.system("echo 21 > /sys/class/gpio/export")
  os.system("echo out > /sys/class/gpio/gpio21/direction")

exitFlag = 0

imageSizePickle = open("imageSize.pickle","rb")
imageSize = pickle.load(imageSizePickle)
newestFrame = None
currentImg = None

def prepImgForCNN(img):
    img= Image.fromarray(np.uint8(img))
    img = img.resize((imageSize, imageSize), Image.NEAREST)   
    img = np.array(img)
    temp = np.zeros((imageSize, imageSize, 1))  
    temp = img[:, :, newaxis]
    return [temp], [[0, 1]]

class CNNWorker (threading.Thread):
   def __init__(self, threadID, queue, rqueue):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.queue = queue
      self.rqueue = rqueue

   def run(self):
      print "Starting thread " , self.threadID
      verifyDetections(self.queue, self.rqueue)
      print "Exiting " , self.threadID

class frameGraber (threading.Thread):
   def __init__(self):
      threading.Thread.__init__(self)
   def run(self):
      global ready
      global newestFrame
      global getNewFrame
      global gotNewFrame
      if readFromFile:
        i = 0
        for filename in sorted(glob.glob(InputImagesFileLocation)):
          if i < 6000 or i % 2 != 0:
            i += 1
            #continue
            pass
          while not getNewFrame:
            time.sleep(0.0001)
          img = np.uint8(scipy.imread(filename, 'L'))

          if blurrImage:
            img = cv2.blur(img,(5,5))
          newestFrame = img
          getNewFrame = False
          gotNewFrame = True
          ready = True
          i += 1
      else:
        cap = cv2.VideoCapture(1)
        while True:
           ret, frame = cap.read()
           if blurrImage:
             frame = cv2.blur(frame,(5,5))
           newestFrame = frame
           ready = True

#Give XY pixel, get value in array
def XY(array, x, y):
  dim = int(math.sqrt(len(array)/128))
  return array[128*dim*y + x*128 : 128*dim*y + x*128 + 128]

outputImg = np.zeros((10,10))
predictedOn = np.zeros((10,10))
def verifyDetections(queue, rqueue):
    
    while not exitFlag:
        queueLock.acquire()
        if not workQueue.empty():
          print "Got one"
          temp = queue.get()
          queueLock.release()

          _y = temp['y']
          _x = temp['x']
          h = temp['h']
          w = temp['w']

          global sess
          global model
          global currentImg

          howMuchLargerCNNBoxThenCascade = 0
          #potentialRegion = currentImg[max(0, _y - howMuchLargerCNNBoxThenCascade):min(currentImg.shape[0], _y+h + howMuchLargerCNNBoxThenCascade), max(0, _x - howMuchLargerCNNBoxThenCascade):min(currentImg.shape[1], _x+w + howMuchLargerCNNBoxThenCascade)]
          potentialRegion = currentImg
          #potentialRegion = np.uint8(scipy.imread("/media/ecestudent/43CBC47963CEF219/DataSets/Test/video7/7_01565.jpg", 'L'))
          
          start = time.time()
          suggestedRegion = prepImgForCNN(potentialRegion)
          feed_dict = {x:suggestedRegion[0], y:suggestedRegion[1], model.keep_prob: 1.0}
          prepTime = time.time() - start
          start = time.time()
          pred, h_pool5, fcWeights, fcBiases = sess.run([model.pred, model.h_pool5, model.W_fc1, model.b_fc1], feed_dict=feed_dict)
          if debug:
            h_pool5 = h_pool5
            pred2 = model.pred
            #plt.imshow(potentialRegion, cmap='gray')
            plt.figure()
            h_pool5 = h_pool5[0,:,:,:]

            
            for i in range(1, 20):
              activations = h_pool5[:,:,i-1:i]
              activations = np.squeeze(activations)
              #plt.subplot(2,10, i)
              #plt.imshow(activations)
            
            global outputImg
            outputImg = np.zeros((10,10))
            temp2 = fcWeights[:,0]
            #embed()
            for xPos in range(outputImg.shape[0]):
              for yPos in range(outputImg.shape[1]):
                sliceTemp2 = h_pool5[xPos,yPos,:]
                sliceTemp = XY(temp2, xPos, yPos)
                if len(sliceTemp) < 128:
                  embed()

                finalSlice = sliceTemp2 * sliceTemp + fcBiases[1]
                outputImg[xPos, yPos] = np.mean(finalSlice)

            plt.imshow(outputImg, cmap='gray')
            #plt.figure()
            #plt.imshow(currentImg)
            #plt.show()
            #embed()
          global predictedOn
          predictedOn = potentialRegion
          predTime = time.time() - start
          if pred[0] == 1:
            temp['timesCNNSaidNo'] += 1
            temp['valid'] = False
          else:
            temp['valid'] = True
         
          queueLock.acquire()
          rqueue.put(temp)
          queueLock.release()
        else:
          queueLock.release()
        time.sleep(0.0001)

queueLock = threading.Lock()
workQueue = Queue.Queue(10000)
resultQueue = Queue.Queue(10000)

#Not Euclidian distance
def dist(x1, y1, x2, y2):
    return math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))

def softmax(z):  
  z = z / 1000
  z_exp = [math.exp(i) for i in z]
  sum_z_exp = sum(z_exp)
  return [round(i / sum_z_exp, 3) for i in z_exp]
  
x = tf.placeholder(tf.float32, shape=(None,) + (imageSize,imageSize,1))
y = tf.placeholder(tf.float32, (None, 2)) 
model = CNNModel(x, y) 
sess = None

def runWithCascade():


      frameGrabThread = frameGraber()
      frameGrabThread.start()

      for i in range(numCNNThreads):
        thread_ = CNNWorker(i, workQueue, resultQueue)
        thread_.start()

      scaleSize = .5
      if writeToFile:
        os.system("rm Result/*")
      carCascade = cv2.CascadeClassifier('classifiers/21Stage.xml')

      if carCascade.empty():
        print "Classifier load error"
        os._exit(1)

      boxQueue = []
      global sess
      i = 0
      with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=numCNNThreads)) as sess:
          sess.run(tf.global_variables_initializer())
          #Load model
          saver = tf.train.Saver()
          saver.restore(sess, 'model7.ckpt')
          totalTime = 0

          frameTime = 0
          try: 
            while True:
              start = time.time()
              #Read image
              global ready
              global getNewFrame
              global gotNewFrame
              while not ready:
                print "Waiting for frame grabber"
                time.sleep(0.0001)
              global currentImg
              global newestFrame
              if readFromFile:
                getNewFrame = True
                while not gotNewFrame:
                  time.sleep(0.0001)
                gotNewFrame = False

              frame = newestFrame
              if not readFromFile:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                currentImg = np.uint8(gray)
              else:
                currentImg = frame
              currentImg = cv2.resize(currentImg, None, fx=scaleSize,fy=scaleSize)

              #Histogram Equilization
              hist,bins = np.histogram(currentImg.flatten(),256,[0,256])
              cdf = hist.cumsum()
              cdf_normalized = cdf * hist.max()/ cdf.max()
              cdf_m = np.ma.masked_equal(cdf,0)
              cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
              cdf = np.ma.filled(cdf_m,0).astype('uint8')
              #currentImg = cdf[currentImg]

              #Detect Regions
              cars = carCascade.detectMultiScale(currentImg, scaleFactor=1.15, minNeighbors=2,flags = 0, minSize=(int(currentImg.shape[0]/5), int(currentImg.shape[0]/5)), maxSize=(int(currentImg.shape[0]),int(currentImg.shape[0])) )
              cars = []
              rect = {}
              rect['x'] = 0
              rect['y'] = 0
              rect['h'] = currentImg.shape[1]
              rect['w'] = currentImg.shape[0]
              cars.append(rect)
              #Increment timesIdle for all boxes
              for temp in boxQueue:
                temp['timesIdle'] += 1
              boxQueue = []
              #Throw regions detected into a queue for the CNN to take care of later
              for (_x,_y,w,h) in cars:
                if not useMultiframeFiltering:
                   temp = {}
                   temp['x'] = _x
                   temp['y'] = _y
                   temp['count'] = 10000
                   temp['timesIdle'] = 0
                   temp['timesCNNSaidNo'] = 0
                   temp['h'] = h
                   temp['w'] = w
                   temp['valid'] = False
                   temp['n'] = 1
                   boxQueue.append(temp)
                   continue

                #If empty queue, just add the region
                if len(boxQueue) == 0:
                   temp = {}
                   temp['x'] = _x
                   temp['y'] = _y
                   temp['count'] = 1
                   temp['timesIdle'] = 0
                   temp['timesCNNSaidNo'] = 0
                   temp['h'] = h
                   temp['w'] = w
                   temp['valid'] = False
                   temp['n'] = 1
                   boxQueue.append(temp)
                   continue

                #Otherwise find the closest match distance wise (if one exists)
                closest = 10000000
                index = 0
                indexOfClosest = -1
                for temp in boxQueue:
                  temp['valid'] = False
                  distance = dist(temp['x'] + temp['w']/2, temp['y'] + temp['h']/2, _x + w/2, _y + h/2)
                  if distance < closest:
                    closest = distance
                    indexOfClosest = index
                  index += 1

                #If the current box is the same as the closest box in the list, replace with the new box and increment count
                if closest < thresh:
                     boxQueue[indexOfClosest]['count'] += 1
                     #If not the first time
                     if boxQueue[indexOfClosest]['n'] > 1:
                       boxQueue[indexOfClosest]['x'] = int(boxQueue[indexOfClosest]['x'] + _x/temp['n'] - boxQueue[indexOfClosest]['xPrev']/temp['n'])
                       boxQueue[indexOfClosest]['y'] = int(boxQueue[indexOfClosest]['y'] + _y/temp['n'] - boxQueue[indexOfClosest]['yPrev']/temp['n'])
                       boxQueue[indexOfClosest]['w'] = int(boxQueue[indexOfClosest]['w'] + w/temp['n'] - boxQueue[indexOfClosest]['wPrev']/temp['n'])
                       boxQueue[indexOfClosest]['h'] = int(boxQueue[indexOfClosest]['h'] + h/temp['n'] - boxQueue[indexOfClosest]['hPrev']/temp['n'])
                     else:
                       #First time
                       boxQueue[indexOfClosest]['x'] = _x
                       boxQueue[indexOfClosest]['y'] = _y
                       boxQueue[indexOfClosest]['h'] = h 
                       boxQueue[indexOfClosest]['w'] = w

  
                     boxQueue[indexOfClosest]['hPrev'] = h
                     boxQueue[indexOfClosest]['wPrev'] = w
                     boxQueue[indexOfClosest]['xPrev'] = _x
                     boxQueue[indexOfClosest]['yPrev'] = _y
                     boxQueue[indexOfClosest]['n'] += 1
                     boxQueue[indexOfClosest]['timesIdle'] = 0
                else: #Otherwise, just add the new box to the end of the queue 
                    temp = {}
                    temp['n'] = 1
                    temp['x'] = _x
                    temp['y'] = _y
                    temp['count'] = 1
                    temp['h'] = h
                    temp['w'] = w
                    temp['timesIdle'] = 0
                    temp['timesCNNSaidNo'] = 0
                    temp['valid'] = False
                    boxQueue.append(temp)

                
              #Delete all boxes that have been idle for expirationTime frames or who have consitently been rejected by cnn
              newBoxQueue = [temp for temp in boxQueue if (temp['timesIdle'] < expirationTime and temp['timesCNNSaidNo'] < CNNSaidNoThreshold)]
              boxQueue = newBoxQueue

              #Send each box in boxQueue who's count value is greater then minCountsToShow to CNN for validation
              sentJobs = 0
              for temp in boxQueue:
                print temp['count']
                if temp['count'] < minCountsToShow:
                  continue  
                sentJobs += 1
                queueLock.acquire()
                workQueue.put(temp)
                queueLock.release()
                continue
              #print "Queue size is", sentJobs
              print "Box queue has ", len(boxQueue)
              #Wait until all the jobs have been proccesed
              while resultQueue.qsize() < sentJobs:
                time.sleep(0.00001)
                         
              carCount = 0  
              currentImg = cv2.cvtColor(currentImg, cv2.COLOR_GRAY2BGR)
              #For each region   
              while not resultQueue.empty():
                try:
                    temp = resultQueue.get()
                    if temp['valid']:
                      #print "Valid Car"
                      carCount += 1
                      if showFrame:
                        #cv2.rectangle(currentImg,(temp['x'],temp['y']),(temp['x']+temp['w'],temp['y']+temp['h']),(255,0,0),2)  
                        cv2.circle(currentImg, (temp['x'] + int(temp['w'] / 2),temp['y'] + int(temp['h'] / 2)) , 10, (0,255,0), 25)
                    else:
                      #print "Not a Valid Car"
                      temp['timesCNNSaidNo'] += 1
                      #cv2.rectangle(currentImg,(temp['x'],temp['y']),(temp['x']+temp['w'],temp['y']+temp['h']),(0,0,0),2)  
                      #cv2.circle(currentImg, (temp['x'] + int(temp['w'] / 2),temp['y'] + int(temp['h'] / 2)) , 10, (0,0,255), 25)
                except:
                      pass
              if carCount > 0:
                if useLED:
                 os.system("echo 1 > /sys/class/gpio/gpio21/value")
                pass
              else:
                if useLED:
                 os.system("echo 0 > /sys/class/gpio/gpio21/value")
                pass

              if showFrame:
                  cv2.imshow('frame',cv2.resize(predictedOn, (500,500) ))
                  cv2.imshow('frame2',cv2.resize(outputImg, (500,500) ))

                  if cv2.waitKey(1) & 0xFF == ord('q'):
                      break
              i += 1
              frameTime = time.time() - start
              totalTime += frameTime
              if i % 10 == 0:
                #print "FPS:", 10.0 / totalTime, '                \r'
                totalTime = 0
              if writeToFile:
               cv2.imwrite('Result/' +str(i).zfill(8) + '.jpg',currentImg)
          except KeyboardInterrupt:
           sess.close()
           os._exit(1)


runWithCascade()

