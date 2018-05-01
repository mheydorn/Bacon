Installation Notes:
You will need python bindings for PIL, opencv, IPython, numpy, tensorflow, etc
Everything needed can be pip installed exepct for the cuda and cudnn libraries (if you want gpu acceleration)
Confirmed to work with tensorflow 1.6 - but any version after 1.4 should be ok
I'm using cuda 9.0.176 Any version that's compatible with your version of tensorflow should be ok
These are all python2 scripts

Instructions:
1. Run readimages.py - You will have to modify it to tell it where your positive and negative samples are.
2. Run train.py - It will create a Best_Checkpoints directory in the current directory where it will save the checkpoints with the highest test accuracy.
   train.py also has features for examining which images were misclassified, as well as restoring from a checkpoint. 

