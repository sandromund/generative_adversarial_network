# Dev Log

26.03.24
* Added GPU support, training much faster now 
* Experimented with model architectures
* Moved to LeakyReLU and added Dropout
* Changed decay of first and second order momentum of gradient
* Smaller batch sizes work better
* Generations are discrete but way to many non-zero values
* Discriminator start with very good f1 score, might be buggy.
* Discriminator learns very fast to predict everything as zero
* Added delay to discriminator training 
* Rounded discriminator output (looks like a fix but loss is now very high)
* Precision generator is allways 1

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

