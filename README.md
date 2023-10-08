# $`\mu`$StyleGAN

This is a simplified PyTorch implementation of StyleGAN2 introduced in [Analyzing and Improving the Image Quality of StyleGAN](https://papers.labml.ai/paper/1912.04958) paper

## Idea
The main purpose is to implement a smaller version of the StyleGAN using another interpolation method and evaluate the quality of result model built by my own. 

This is a study project to understand what is GANs, how they can be designed and how can be trained. 

The code is based on https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master/labml_nn/gan/stylegan

## Tools used
- PyTorch
- Numpy
- scikit-learn
- matplotlib (just to visualize training results)

## Dataset
- Santiel 5K Truecolor [Landshapes-4041](https://www.kaggle.com/datasets/ueberf/sentinel-5k-truecolor)

## Issues
- Because of small computational power the model can't be trained longer to achieve desired image quality
- Currently looking for an opportunity to run model in distributed mode (multiple gpu's/multiple nodes/combined)
- A simple web-based API for image generation is currently unimplemented.

## The main differences from original StyleGAN
- The network generates 32 $\times$ 32 px instead of 1024 $\times$ 1024 px.
- Smaller architecture: mapping network (5 layers used instead of 8 in original), generator and discriminator ($\log_2(32) = 5$ blocks instead of $\log_2(1024) = 10$)
- Noise injection was performed in two steps:
   - 1. Generate the base noise before mapping styles.
   - 2. In each style block add randomly generated noise instead of using pregenerated outter.
 - Default interpolation method is `nearest-exact` instead of `bilinear` so produced images are more smooth.

# Results

## Loss

|![Loss](images/loss.jpg)|
|:--:|
|Image 1 - Generator and discriminator train losses per epochs|

## Generated samples

| ![Samples](images/samples.jpg) |
| :--: |
| Image 2 - training results per epochs: 1200 epochs (a), 1500(b), 3200(c), 3500(c), 4200(e)|


## Generated and original images comparison

|![Comparison](images/comparison.jpg)|
|:--:|
|Image 3 - Comparison of zoomed generated (left) and real (right) images|
