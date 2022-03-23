# Re-implementation of Multiple Color-space Fusion Network (MCF-Net)
MCF_Net is a deep-learning system based on *DenseNet121* for retinal image quality assessment (RIQA) developed by [Fu et al (2019)](https://arxiv.org/abs/1907.05345). Original project web can be found [here](https://github.com/HzFu/EyeQ).  

-----------------
*Huazhu Fu, Boyang Wang, Jianbing Shen, Shanshan Cui, Yanwu Xu, Jiang Liu, Ling Shao, "Evaluation of Retinal Image Quality Assessment Networks in Different Color-spaces", in MICCAI, 2019.*

-----------------
## Brief Description of Network Architecture
MCF-Net transforms RGB retinal images into two other colour spaces, i.e. HSV and LAB, before passing all three independently to different base CNNs. The output of these base networks are then concatenated within a fusion block (**feature-level** fusion). The output of this feature-level fusion is then concatenated with the output of the three independent base networks (**prediction-level** fusion). The authors argue that this two-level fusion 'guarantees the full integration of the different colour spaces'. MCF-Net achieved a test accuracy of **88.00%.**

## Eye-Quality Dataset (Eye-Q)
The network was built on Eye-Q, a subset of the [EyePACS](https://www.kaggle.com/c/diabetic-retinopathy-detection) dataset re-annotated with labels describing image quality as follows:

**GOOD:** *image quality is high enough for diagnosis of general pathology, e.g. diabetic retinopathy, age-related macular degeneration, glaucoma. As such, optic nerve head, macula and retinal vessels should be sufficiently visible.*

**USABLE:** *presence of slight low-quality indicators but structures with diagnostic value still identifiable by clinicians.*

**REJECT:** *full and reliable diagnosis not possible due to serious quality issue.* 

## Re-implementation Steps
1. Create a new folder in home directory and upload retinal images to that folder.
2. **Download** the trained model (DenseNet121_v3_v1.tar; 28.86M parameters) [here](https://onedrive.live.com/?authkey=%21AJEJujrShK2M9Zk&cid=F3A8A31ABFAC51B0&id=F3A8A31ABFAC51B0%213790&parId=F3A8A31ABFAC51B0%21253&action=locate) and **upload** it to 'MCF_Net'.
3. Run MCF_Net/test.py from terminal: *python MCF_Net/test.py --test_images_dir {name of image folder} --res_name {name of result folder}*
4. Result file *(res_name)* can be found in home directory.

## Examples of Images Predicted as 'Good', 'Usable' and 'Reject'
*Images are taken from EyePACS 'sample.zip'*

**GOOD      —>      USABLE      —>      REJECT**
<p float="left">
  <img src="/sample_images/good.jpeg" width="200" /> 
  <img src="/sample_images/usable.jpeg" width="200" /> 
  <img src="/sample_images/reject.jpeg" width="200" />
</p>


