Installation Notes:
1. You will need python bindings for PIL, opencv, IPython, numpy, tensorflow (or tensorflow-gpu), etc
2. Everything needed can be pip installed exept for the cuda and cudnn libraries (if you want gpu acceleration)
3. Confirmed to work with tensorflow 1.6 but any version after 1.4 should be ok
4. I'm using cuda 9.0.176 but any version that's compatible with your version of tensorflow should be ok
5. These are all python2 scripts

Instructions:
1. Run readimages.py - You will have to modify it to tell it where your positive and negative samples are.
2. Run train.py - It will create a Best_Checkpoints directory in the current directory where it will save the checkpoints with the highest test accuracy.
   train.py also has features for examining which images were misclassified, as well as restoring from a checkpoint. 

Progress / Further Work:
1. I haven't done a lot with tweaking the architecture or hyperparameters (just learning rate and drop out rate in this case). There's probably a lot to be gained there
2. It gets up to 80% accuracy without any data augmentation, drop out keep rate set to 0.5,  and a learning rate of 0.5e-4
3. With random flipping across x,y, and both axis we're at 86%
  

