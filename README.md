# Thesis20-Models
Two contender models for our Filipino Sign Language Recognition application trained on three sign categories (Greetings, Colors, Days) from the FSL-105 dataset by (Tupal, 2023) and additional videos added by us, the researchers. 

This repository contains two folders for the two contender models:
   <br>&ensp;(1) Model-1-97_  (97% Testing Accuracy)
   <br>&ensp;(2) Model-2-94_  (94% Testing Accuracy)

   &ensp;Each folder contains:
      <br>&ensp;&ensp;(1) Six .npy files for the test-train-val sets (We also have a version where both models were trained and evaluated on the same test-val-test sets, but currently, we've decided to upload the version trained/eval'ed on different sets first)
      <br>&ensp;&ensp;(2) One .ipynb file which contains the training and evaluation runtime.
      <br>&ensp;&ensp;(3) One .keras file which contains the model weights.

**Reference**
Tupal, Isaiah Jassen (2023), “FSL-105: A dataset for recognizing 105 Filipino sign language videos”, Mendeley Data, V2, doi: 10.17632/48y2y99mb9.2
