---------------------------------
PROJECT DEPENDENCIES:
matpoltlib
tensorflow-gpu
tensorflow-hub
zipfile
datetime
---------------------------------
HOW TO GET THE DATASET
The dataset is quite large so I used LFS to store the file in the repo.
If you have LFS installed you will just need to "pull LFS" files from your git client.
If you do not have LFS installed you can download the data set from here - https://drive.google.com/file/d/1WH-aEQhdJn7VUyoAWdwWibubl19o0jEP/view?usp=sharing
Once you have the data set .zip folder, make sure it is in the "Image Recognition/Data/" folder.
After that, you just extract the zip folder in the Data folder and you are all set.
---------------------------------
HOW TO START THE NEURAL NETWORK
There are 2 .py scripts in the "Image Recognition" folder.
One is the Base neural network without transfer learning.
The other is with transfer learning.
They're completely different so I thought they could both go in the folder.
---------------------------------