# VAE

**Goal: Learn how VAEs work**

### Simple VAE explanation
A variational autoencoder tries to encode the input data to a latent space much smaller than the original input (e.g. Pokemon input image is 128x128x3, latent space is 256), from which it then tries to decode it back into the original image. By trying to do so, it learns to create this latent space so that it fits the most important information it needs. It is then possible to travel through the latent space and change aspects of the input image or try to generate new images.


### Data
The [data](https://github.com/RiccardoGrin/VAE/tree/master/Pokemon) for the Pokemon was taken from the PokeGAN repository that I worked with previously. Other data, such as MNIST can be used with the network by easily adding it to the dataloader, and giving it a name for it to load (will be obvious once the dataloader is opened).


### Training & Results
During training it starts by creating really blurry blob like shapes of the pokemon, but quickly learns the shape within the first 100 or so epochs (not much datta, therefore many epochs). However it then takes another about 200 epochs to get the colour correct for most Pokemon. The colours that it struggles with the most are green, pink, and white, which even after 1500 epochs, Pokemon of mainly these colours are difficult to reproduce correctly.

These are the genertaion results on the 1st, 500th, and 1000th epoch:
<p pad="10" align="center"> 
<img src="https://github.com/RiccardoGrin/VAE/blob/master/resources/img_0001.png" width="200" height="200" hspace="20"/>
  <img src="https://github.com/RiccardoGrin/VAE/blob/master/resources/img_0500.png" width="200" height="200" hspace="20"/>
  <img src="https://github.com/RiccardoGrin/VAE/blob/master/resources/img_1000.png" width="200" height="200" hspace="20"/>
</p>

This is a gif compilation of training over 1500 epochs:
<p pad="10" align="center"> 
<img src="https://github.com/RiccardoGrin/VAE/blob/master/resources/pokemon_gen.gif" width="200" height="200" hspace="20"/>
</p>

Here are some other examples (top: original - bottom: reproduced), which can be made through the notebook, once the model is trained:
<p pad="10" align="center"> 
<img src="https://github.com/RiccardoGrin/VAE/blob/master/resources/inference.png" hspace="20"/>
</p>



### Demo
With the data already part of the repository, the model can be easily trained from scratch, either through the Jupyter Notebook file, or through the Python train file. To view multiple images in a plot after training, use the notebook.

```
python train.py
```
