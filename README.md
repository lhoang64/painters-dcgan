
# paintersDCGAN

For my senior work at Bennington College, I build a DCGAN model following the specfications of the paper [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) using Keras+Tensorflow. 

The model was trained using the [Painter by Numbers](https://www.kaggle.com/c/painter-by-numbers) dataset.

In this project I aimed to test its effectiveness in generating paintings when trained on artwork classified by artistic style. Artwork classified by styles because with artwork, a class can be contested and people often debate over art classification systems. It is also difficult to define what to look for in a painting to be able to determine what style is it.

## The Dataset
Using images and labels from the Painter by Numbers dataset, I organized the images by style label. The model was then trained on labeled images.
The dataset contains 135 style labels, with the 10 most represented styles are:


## Results
 In my experiments, I employed the DCGAN architecture on different style datasets. Despite trying different recommended optimizations, I did not obtain results that resembled the input data. This could be due to the model's architecture, the training dataset lacking enough identifiable features, or hardware limitations. 

 However I was able to achieve visually interesting results that also exhibitted learning. The following image show a compilation of the results obtained from each iteration of the DCGAN model. Styles represented here include Abstract Expressionism, Impressionism, Minimalisn, New Casualism, and Realism.


 A full discussion of my experiments and results can be found [here](https://docs.google.com/document/d/1SaE1gjaPsunO-TerIAyPAp1w7C1d7f7j6YSGwlkd3ts/edit?usp=sharing).