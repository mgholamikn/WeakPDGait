# PD_Gait_labeling
Implementation of "Automatic Labeling of Parkinson’s Disease Gait
Videos with Weak Supervision". 

<p align="center">.
<img  src="Figures/tiser.jpg" width="800">
<p/>

# Environment
Required packages:
<ul>
  <li>pytorch</li>
  <li>pytorch3d</li>
</ul> 

To install pytorch3d please follow the instructions at https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md

# 3D Humna Pose 

Training on Human3.6M dataset. Please first download the data from here and put in the ```/data``` directory.
```
python train.py
```

# PD Labeling
```
python train_PD.py
```
