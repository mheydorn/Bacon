'''
Reads in the image filenames and creates a datastructure for dataset.py to read.
This is also where the test/train split happens
'''

from PIL import Image
import glob
import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy import ndimage
from skimage.measure import block_reduce

imageSize = 96

'''
Lists (one for negatives and one for positives) where each item in the list is a list of
length 2 containing first the image filename and second the label in one hot encoding
'''
images_and_labels_positive = []
images_and_labels_negative = []

#Positives
for filename in glob.glob('bacon_split/1/*.png'):  
  temp = []
  temp.append(filename)
  temp.append([1, 0])
  images_and_labels_positive.append(temp)

#Negatives
for filename in glob.glob('bacon_split/0/*.png'):
  temp = []
  temp.append(filename)
  temp.append([0, 1])
  images_and_labels_negative.append(temp)

print "Saving to disk"
images_and_labels_positive_pickle = open("images_and_labels_positive.pickle","wb")
pickle.dump(images_and_labels_positive, images_and_labels_positive_pickle)

images_and_labels_negative_pickle = open("images_and_labels_negative.pickle","wb")
pickle.dump(images_and_labels_negative, images_and_labels_negative_pickle)

size_pickle = open("imageSize.pickle","wb")
pickle.dump(imageSize, size_pickle)

print "Found", len(images_and_labels_positive), "positive images"
print "Found", len(images_and_labels_negative), "negative images"
print "Done"
