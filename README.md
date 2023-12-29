# Automatic Labeling of Parkinson’s Disease Gait Videos with Weak Supervision
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GBSTHjqCJH60B0J_Ol3Zih8vTAFW_ptS?usp=sharing)


Implementation of <a href="https://www.sciencedirect.com/science/article/pii/S1361841523001317">"Automatic Labeling of Parkinson’s Disease Gait
Videos with Weak Supervision"</a>. 

<p align="center">.
<img  src="Figures/tiser.jpg" width="800">
<p/>

# Environment
Required packages:
<ul>
  <li>pytorch</li>
  <li>pytorch3d</li>
  <li>snorkel</li>
</ul> 

To install pytorch3d please follow the instructions at <br>https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md
<br>To install snorkel please follow the instructions at <br>https://www.snorkel.org/get-started/ 

# 3D Human Pose 

First, train the 3D pose estimator on the Human3.6M dataset. Please first download the data from <a href="https://drive.google.com/drive/folders/1YnIYQldiPAphX3gI4yzmbKPeSL_kiD6p?usp=share_link">here</a> and put in the ```/data``` directory.
```
python train.py
```

Then you need to fine-tune the pose estimator on your multi-view data. We can not provide our PD data due to privacy issues. You can use your data to train the network. Instructions on preparing custom data will be added.
```
python train_PD.py
```

Citation:

```
@article{GHOLAMI2023102871,
title = {Automatic labeling of Parkinson’s Disease gait videos with weak supervision},
journal = {Medical Image Analysis},
volume = {89},
pages = {102871},
year = {2023},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2023.102871},
url = {https://www.sciencedirect.com/science/article/pii/S1361841523001317},
author = {Mohsen Gholami and Rabab Ward and Ravneet Mahal and Maryam Mirian and Kevin Yen and Kye Won Park and Martin J. McKeown and Z. Jane Wang},
}
```
