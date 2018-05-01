'''
Script to train on the images proccesed by readImages.py

It will evaluate test accuracy after every negative epoch. This makes it a bit slower but is neccesary in order
determine when overfitting starts to happen. Due to the large size of the epochs, it may even be helpfull to evaluate 
test accuracy during a training epoch to more precisely determin when overfitting starts.

An epoch in this case is when all of the negative images (do not have foregn objects) have been passed through the network.
Since an even number of positive and negative images are given to the network at a time and there are far fewer
positive images, each positive image will go through the network many times per epoch. This is why the ouput says "Finished 
positive epoch" many times before finishing a complete epoch.

It is not uncommon for the test accuracy to be a little higher then the train accuracy because the train accuracy is an average
of the accuracies while training, meaning the earlier accuracies weren't trained as far as the later ones. The test accuracy 
however is evaluated after all of the training has completed for the epoch, which gives it the advantage.

'''

from __future__ import absolute_import
from __future__ import division

import os
import sys
import csv
import json
import argparse
import matplotlib.pyplot as plt
from IPython import embed
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import scipy.misc as scipy
from PIL import Image
import numpy as np
import tensorflow as tf
import math
import dataset
from model import CNNModel
import pickle
import math

'''
TODO:
If you restore from a checkpoint this scipt will not know anything about previous accuracies 
which will cause it to write the first checkpoint it gets to into the Best_Checkpoints directory thinking 
that the first test accruacy it gets is the best so far. To solve this, we could save the best accuracy
so far to disk and load it at the same time that we load the checkpoint file. It will also think you are 
starting from epoch 0 since the epoch count is not saved to disc either. This could be solved a similar way.
However, the short term solution is to just do all of the training at once and only use restoreFromCheckpoint 
when evaluating a checkpoint in more detail later on (for example when you want to examine which images the network
missed)
'''
restoreFromCheckpoint = False
checkpointFile = 'model24.ckpt'

'''
If set to True, writeResults shows which images were missed and which were not missed in a test epoch. 
If set to True, you will probably want to just run one test epoch combined with setting restoreFromCheckpoint to True
It should overwrite the results (images) from the previouse epoch otherwise with the results from whichever checkpoint
was run last (regardless of wither is was most accurate or not)
'''
writeResults = False
if writeResults:
  os.system("mkdir Results")
  os.system("mkdir Results/Negatives")
  os.system("mkdir Results/Positives")
  os.system("mkdir Results/Negatives/Correct")
  os.system("mkdir Results/Negatives/Incorrect")
  os.system("mkdir Results/Positives/Correct")
  os.system("mkdir Results/Positives/Incorrect")

os.system("mkdir Best_Checkpoints")

imageSizePickle = open("imageSize.pickle","rb")
imageSize = pickle.load(imageSizePickle)

def train():
    #Load dataset
    data = dataset.read_data_sets()

    obs_shape = (imageSize,imageSize, 3)
    num_class = 2
    x = tf.placeholder(tf.float32, shape=(None,) + obs_shape)
    y = tf.placeholder(tf.float32, (None, num_class))
    model = CNNModel(x, y)

    ''' 
    The batch size is the number of images given to the network at a time. The more memory 
    your gpu has the larger you can make your batch size, which can effect train time.
    The network is given batchSize/2 positive and batchSize/2 negative images. Since we 
    consider an epoch a complete pass of the negative images, you will only move trough 
    an epoch at batchSize/2 intervals
    '''
    batchSize = 100
    numEpochs = 2000
    learningRate = 0.5e-4

    optimizer = tf.train.AdamOptimizer(learningRate).minimize(model.loss)
    saver = tf.train.Saver(tf.trainable_variables())
    bestTestAccuracySoFar = 0
    with tf.Session() as sess:
        print('Starting training')

        sess.run(tf.global_variables_initializer())
        if restoreFromCheckpoint:
          saver.restore(sess, checkpointFile)
        for epoch in range(numEpochs):

            begin = time.time()

            print "Starting epoch", epoch
            #Training
            train_accuracies = []
            while True:
                batch = data.train.next_batch(batchSize, flip = True)  
                feed_dict = {x:batch[0], y:batch[1], model.keep_prob: 0.5}
                _, acc = sess.run([optimizer, model.accuracy], feed_dict=feed_dict)
                train_accuracies.append(acc)
                #If done with epoch
                if batch[2]:
                  break
                
            print "Finished training for epoch ", epoch
            print "Evaluating test accuracy..."

            batchNum = 0
            test_accuracies = []
            #Testing
            while True:
                batch = data.test.next_batch(batchSize)  
                #We do not pass in an optimizer to the run function and we set the keep_prob to 1.0 here because we are only evalutaing
                feed_dict = {x:batch[0], y:batch[1], model.keep_prob: 1.0}
                correct, acc = sess.run([model.correct_prediction, model.accuracy], feed_dict=feed_dict)
                test_accuracies.append(acc)
                index = 0
                if writeResults:
                  for negorpos, img, isCorrect in zip(batch[1], batch[0], correct):
                    #If negative
                    if negorpos[1] == 1:
                      if isCorrect:
                        cv2.imwrite("Results/Negatives/Correct/" + str(batchNum) + "_" + str(index) + ".jpg", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                      else:
                        cv2.imwrite("Results/Negatives/Incorrect/" + str(batchNum) + "_" + str(index) + ".jpg", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    else: #Else positive
                      if isCorrect:
                        cv2.imwrite("Results/Positives/Correct/" + str(batchNum) + "_" + str(index) + ".jpg", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                      else:
                        cv2.imwrite("Results/Positives/Incorrect/" + str(batchNum) + "_" + str(index) + ".jpg", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    index += 1
                batchNum += 1
                #If done with epoch
                if batch[2]:
                  break 

            train_acc_mean = np.mean(train_accuracies)
            test_acc_mean = np.mean(test_accuracies)

            print "Epoch=", epoch, ", time =", time.time()-begin, ", train accuracy=", train_acc_mean, "test accuracy=", test_acc_mean

            if(test_acc_mean > bestTestAccuracySoFar):
              bestTestAccuracySoFar = test_acc_mean
              saver.save(sess, "./Best_Checkpoints/model_with_test_accuracy_" + str(test_acc_mean) +"_.ckpt")
              print "Saved checkpoint to Best_Checkpoints directory since this epoch had the best test accuracy so far"
            else:
              print "Not saving checkpoint to disk since its test accuracy was not the best so far"
        sess.close()
        os._exit(1)

train()










