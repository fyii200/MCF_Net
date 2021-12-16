# Multiple Color-space Fusion Network (MCF-Net)
Re-implementation of MCF_Net, a deep learning model based on *DenseNet121* for retinal image quality assessment (RIQA) developed by [Fu et al (2019)](https://arxiv.org/abs/1907.05345). Original project web can be found [here](https://github.com/HzFu/EyeQ).  

-----------------
## Brief Description of Network Architecture
MCF-Net transforms RGB retinal images into two other colour spaces, i.e. HSV and LAB, before passing all three independently to different base CNNs. The output of these base networks are then concatenated within a fusion block, representing **feature-level** fusion. The output of this feature-level fusion is then concatenated with the output of the three independent base networks. This second fusion corresponds to **prediction-level** fusion. The authors argue that this two-level fusion 'guarantees the full integration of the different colour spaces'. MCF-Net achieved a test accuracy of **91.75%.**

## Eye-Quality Dataset (Eye-Q)
The network was built on Eye-Q, a subset of the [EyePACS](https://www.kaggle.com/c/diabetic-retinopathy-detection) dataset re-annotated with labels describing image quality as follows:

**GOOD:** *image quality is high enough for diagnosis of general pathology, e.g. diabetic retinopathy, age-related macular degeneration, glaucoma. As such, optic nerve head, macula and retinal vessels should be sufficiently visible.*

**USABLE:** *presence of slight low-quality indicators but structures with diagnostic value still identifiable by clinicians.*

**REJECT:** *full and reliable diagnosis not possible due to serious quality issue.* 

## Re-implementation Steps
1. upload images to the 'images' folder found in home directory.
2. Run MCF_Net/test.py script. Alternatively, run implementation.ipynb in home directory.
3. Result file can be found in home directory.

## Examples of Images Predicted as 'Good', 'Usable' and 'Reject'
*Images are from EyePACS, under 'sample.zip'.*

[Good](https://user-images.githubusercontent.com/72454128/146388468-a9bb8189-51e7-47dc-b807-9cdef089c79f.jpeg)
[Usable](https://user-images.githubusercontent.com/72454128/146388930-ac97f718-129e-42fc-a30e-278c3dc838bc.jpeg)
[Reject](https://user-images.githubusercontent.com/72454128/146388958-7db0a2b8-30c7-4deb-a3c3-7fdf28e054ab.jpeg)
