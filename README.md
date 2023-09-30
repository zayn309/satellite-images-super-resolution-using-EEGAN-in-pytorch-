# satellite images super resolution using EEGAN and EESRGAN in pytorch
## Model Architecture 
### EEGAN
![image](https://github.com/zayn309/satellite-images-super-resolution-using-EEGAN-in-pytorch-/assets/102887305/db6a2df7-dc16-4e39-b762-4825848a1bf7)
### EESRGAN
![image](https://github.com/zayn309/satellite-images-super-resolution-using-EEGAN-in-pytorch-/assets/102887305/2e348d63-5451-41d7-9724-c2369b316bce)

## clarifications about the model achitecture
the achitecture for the two models is the same for the discriminator, but the change is in the generatores as for the EEGAN i used a dense block for the feature extraction and for the EESRGAN i replaced the dense blocks with RRDB (residual in residual dense block) for both feature extraction and Edge enhancement, which raised better results and helped more with the vanishing gradiants.

## Dependencies and Installation
navigate to the directory where the requirements.txt exists and run the command ```pip install -r requirements.txt ``` to install the packages.
> **Note**
> You're also gonna need an NVIDIA GPU + CUDA for the training, altho the inference can be done on the cpu but it's gonna be slow.
## metrics
#### PSNR - ```peak signal-to-noise ratio```
![image](https://github.com/zayn309/satellite-images-super-resolution-using-EEGAN-in-pytorch-/assets/102887305/64b98e28-db35-4a04-9985-95952854b002)
#### SSIM - ```Structural Similarity Index``` 
![image](https://github.com/zayn309/satellite-images-super-resolution-using-EEGAN-in-pytorch-/assets/102887305/473ed806-754f-4724-8ea1-e4a99159c045)
> **Note**
> As the images is multichannel, so these measures is calculated across all channels and then summed and normalized by the number of channels

## Dataset
the dataset is from NARSS (National Authority for Remote Sensing and Space Sciences), and it consists of a single image with resolution 4648 x 4242 x 4 where 4 is the number of channels which are RGB and near infrared, then i cropped into smaller batches of shape 256 x 256 x 4, then applied bicubic interpolation to reduce the resolution of the images by a scale factor of 4 to get the low resolution images, and the shape of the low resolution images is 64 x 64 x 4. 
The data is then scaled using this script 
```python
def scaleCCC(x):
    x = np.where(np.isnan(x), 0, x)
    return((x - np.nanpercentile(x, 2))/(np.nanpercentile(x, 98) - np.nanpercentile(x,2)))
```
> **Note**
> Not using the max value for the scaling but instead 98th percentile ensures clipping the outliers which can ruin the scalling hence the training.

## Loss
I used a combination of multible losses to produce the final objective which consists of:
* ``` Content Loss ```  applying the loss on the pixel space only would be inconvenient, as the images is not only pixels but also the features that is constituted by these pixels,so we need to bring the image into some feature space in order to get effective loss, which done by the Content Loss by propagating the image through the VGGnet to get the features of the images then applying the charbonnier loss P on these features of the image and the ground truth to get the loss where ``` P(x) = (x 2 + ε2)^(1/2) ```
> **Important**
> As our dataset consists of 4 banded images, and the VGGnet accepts only RGB images, we had to apply PCA on the images to make channel reduction from 4 to 3 channels in order to be able to propagate it through the VGGnet without lossing much information.
* ``` consistency Loss ```  the consistency loss is computed by applying the charbonnier loss across the pixels.
> Note: Not that the pixel wise loss is useless, it's important to keep the consistency between our target image and the ground truth.
* ``` adversarial loss ```  which pushs the discriminator to identify fake image and real images hence it pushs the generator to produce more realistic images, and can be computed as
``` Lossadv(θD) = −logD(IHR) − log(1 − D(G(ILR))) ```

So our final objectiv is produced by the formula ``` L(θG, θD) = LOSScont(θG) + αLOSSadv(θG, θD)+λLOSScst(θG) ```  where α and λ are the weight parameters to balance the loss
components. I empirically set the weight parameters α and λ to 1 × 10−3 and 5.

## Training
``` python train.py ```

## Results
low resolution image       |  Super Resolved Image
:-------------------------:|:-------------------------:
![image](https://github.com/zayn309/satellite-images-super-resolution-using-EEGAN-in-pytorch-/assets/102887305/8900b080-5a79-4bf0-8bda-c7f8bd7cf3e9)  |  ![image](https://github.com/zayn309/satellite-images-super-resolution-using-EEGAN-in-pytorch-/assets/102887305/0e4fcf54-6d0e-414e-9900-b0cf1b64e962)


|PSNR|SSIM|
|--|--|
|<table> <tr><th>epoch</th><th>result</th></tr><tr><td>1</td><td>22.1727</td><tr><td>10</td><td>29.7129</td><tr><td>20</td><td>30.3184</td><tr><td>30</td><td>30.5289</td></tr> </table>| <table> <tr><th>epoch</th><th>result</th></tr><tr><td>1</td><td>0.6554</td></tr><tr><td>10</td><td>0.8265</td></tr><tr><td>20</td><td>0.8395</td></tr><tr><td>30</td><td>0.8423</td></tr> </table>|

The model raised good results after only 30 epochs of training, altho it can get better maybe by getting more data and applying some data augmentations.

## Paper
Find the paper of the EESRGAN [here](https://arxiv.org/abs/2003.09085)

and the paper for the EEGAN  [here](https://www.researchgate.net/publication/332089421_Edge-Enhanced_GAN_for_Remote_Sensing_Image_Superresolution)
