# Dev Log

26.03.24
* Added GPU support, training much faster now 
* Generations are discrete but way to many non-zero values
* Maybe other loss function?
* Moved to LeakyReLU

14.03.24

* Tried one-hot-encoding predictions
* Discrete GAN predictions are an open
  problem [see](https://stats.stackexchange.com/questions/533641/how-do-gans-handle-discrete-outputs)
* Found Papers that might
  help: [DWGAN](https://openreview.net/pdf?id=Bkv76ilDz), [DGSAN](https://arxiv.org/pdf/1908.09127.pdf)
* Experimenting with model architectures and hyperparameters 
* Generator is better now

13.03.24

* First version is running on real data
* Discriminator outperforms Generator way to fast
* Generations not that good

