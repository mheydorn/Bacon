import pickle
import matplotlib.pyplot as plt
import numpy as np
import pickle
from PIL import Image
from IPython import embed
from PIL import Image
import glob
import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy import ndimage
from skimage.measure import block_reduce
import random
from random import shuffle
import cv2

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

random.seed(32423)

imageSizePickle = open("imageSize.pickle","rb")
imageSize = pickle.load(imageSizePickle)

'''
Randomely does one of 4 possibilities: flips horizontally, flips vertically, flips over both axis, does nothing
'''
def flipImage(img):
  #Get random int between in interval (-1,2)
  flipType = random.randint(-1,3)
  #1 in 4 change we don't flip
  if flipType == 2:
    return img
  return cv2.flip(img, flipType)

def filenamesToImages(filenameList, flip):
  output = []  
  for filename in filenameList:
    img=(Image.open(filename))
    img = np.array(img.resize((imageSize, imageSize), Image.NEAREST))
    if flip:  
      img = flipImage(img)
    output.append(img)
  return output

def extractColumn(list2D, col):
  return [row[col] for row in list2D]

class DataSet(object):
  def __init__(self, images_positive, labels_positive, images_negative, labels_negative):

    self._num_examples_positive = len(images_positive)
    self._num_examples_negative = len(images_negative)
    self._images_positive = images_positive
    self._labels_positive = labels_positive
    self._images_negative = images_negative
    self._labels_negative = labels_negative
    self._epochs_completed = 0
    self._epochs_completed_negative = 0
    self._index_in_epoch_positive = 0
    self._index_in_epoch_negative = 0
    self.batches_proccesed = 0

  '''
  This is where all the logic happens to feed an ever number of positive and negative images to the network at the same time.
  This is also were it is determined if an epoch has been completed. 
  '''
  def next_batch(self, batchSize, flip = False):
    batchSize = int(batchSize / 2)

    #Show progress every 100 batches
    if self.batches_proccesed % 100 == 0:
      print "######### At image ",  self._index_in_epoch_negative, "of epoch", self._epochs_completed_negative,"##########"
    self.batches_proccesed += 1

    start_positive = self._index_in_epoch_positive
    start_negative = self._index_in_epoch_negative
    epochDone = False

    #If we don't have enough samples to finish another training batch for positives then we have to rollover
    if start_positive + batchSize > self._num_examples_positive:
      print "Finished positive epoch"
      # Finished epoch
      self._epochs_completed += 1
      # Get remaining examples in this training batch
      rest_num_examples = self._num_examples_positive - start_positive
      images_rest_part = self._images_positive[start_positive:self._num_examples_positive]
      labels_rest_part = self._labels_positive[start_positive:self._num_examples_positive]
      # Start next epoch
      self._index_in_epoch_positive = batchSize - rest_num_examples
      end = self._index_in_epoch_positive
      self._index_in_epoch_positive = 0
      images_new_part = self._images_positive[0:end]
      labels_new_part = self._labels_positive[0:end]

      positive_images =  np.concatenate((images_rest_part, images_new_part), axis=0)
      positive_labels = np.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch_positive += batchSize
      end = self._index_in_epoch_positive

      #The case when we have exactle the right number of images left in the epoch to complete the batch
      if self._index_in_epoch_positive == self._num_examples_positive:
        print "Finished positive epoch"
        start_positive = 0
        self._index_in_epoch_positive = 0
        self._epochs_completed += 1
        positive_images =  self._images_positive[self._num_examples_positive - batchSize: self._num_examples_positive]
        positive_labels = self._labels_positive[self._num_examples_positive - batchSize: self._num_examples_positive]     
      else:
        positive_images =  self._images_positive[start_positive:end]
        positive_labels = self._labels_positive[start_positive:end]


    #If we don't have enough samples to finish another training batch for negatives then we have to rollover
    if start_negative + batchSize > self._num_examples_negative:
      epochDone = True
      print "Completed all negatives"
      # Finished epoch
      self._epochs_completed_negative += 1
      # Get remaining examples in this training batch
      rest_num_examples = self._num_examples_negative - start_negative
      images_rest_part = self._images_negative[start_negative:self._num_examples_negative]
      labels_rest_part = self._labels_negative[start_negative:self._num_examples_negative]
      self._index_in_epoch_negative = batchSize - rest_num_examples
      end = self._index_in_epoch_negative
      self._index_in_epoch_negative = 0
      images_new_part = self._images_negative[0:end]
      labels_new_part = self._labels_negative[0:end]  
      negative_images =  np.concatenate((images_rest_part, images_new_part), axis=0)
      negative_labels = np.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch_negative += batchSize
      end = self._index_in_epoch_negative

      #The case when we have exactle the right number of images left in the epoch to complete the batch
      if self._index_in_epoch_negative == self._num_examples_negative:
        epochDone = True
        print "Completed all negatives"
        self._epochs_completed_negative += 1
        start_negative = 0
        self._index_in_epoch_negative = 0
        negative_images =  self._images_negative[self._num_examples_negative - batchSize: self._num_examples_negative]
        negative_labels = self._labels_negative[self._num_examples_negative - batchSize: self._num_examples_negative] 
      else:    
        negative_images =  self._images_negative[start_negative:end]
        negative_labels = self._labels_negative[start_negative:end]

    positive_images = filenamesToImages(positive_images, flip)
    negative_images = filenamesToImages(negative_images, flip)

    return np.concatenate((positive_images, negative_images), axis=0), np.concatenate((positive_labels, negative_labels), axis=0), epochDone




'''
Creates a tensorflow Datasets object from pickled files created by readImages.py
This is also where the datasets are split and shuffled
'''
def read_data_sets():      

  print "Loading positives from disk"
  pickleIn = open("images_and_labels_positive.pickle","rb")
  positives = pickle.load(pickleIn)

  print "Loading negatives from disk"
  pickleIn = open("images_and_labels_negative.pickle","rb")
  negatives = pickle.load(pickleIn)

  positives_train = positives[0:int(len(positives)*.7)]
  negatives_train = negatives[0:int(len(negatives)*.7)]

  positives_test = positives[int(len(positives)*.7)+1 : int(len(positives))-1]
  negatives_test = negatives[int(len(negatives)*.7)+1 : int(len(negatives))-1]

  '''
  The images from which the patches were extracted overlapped, meaning we have to shuffle after
  the train/test split in order to avoid having similar objects in the test and train images, 
  which would be cheating.

  '''
  print "Shuffling data"
  #Note that shuffle works in place and returns None
  shuffle(positives_train)
  shuffle(negatives_train)
  shuffle(positives_test)
  shuffle(negatives_test)

  print "Creating train dataset object"
  train = DataSet(extractColumn(positives_train, 0), extractColumn(positives_train, 1), extractColumn(negatives_train, 0), extractColumn(negatives_train, 1))

  print "Creating test dataset object"
  test = DataSet(extractColumn(positives_test, 0), extractColumn(positives_test, 1), extractColumn(negatives_test, 0), extractColumn(negatives_test, 1))

  return base.Datasets(train=train, validation=None, test=test)




















