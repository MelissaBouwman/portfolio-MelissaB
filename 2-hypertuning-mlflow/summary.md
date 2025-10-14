# Summary week 2

Hypothesis: I expected that adding a dropout would reduce the overfitting problem, especially in bigger networks with more filters/dense units. Therefore, i also expected that batch normalization would stabilise the training process and would speed up the convergence. WHile using more convolational layers i expected better validation results, this because the model can learn hierarchical feautures. 

Results:

I used the FASHION-MNIST dataset with a batchsize of 64. Mainly i experimented with the following hyperparameters:
- Dropout -> 0.0 0.2 0.5
- Normalization -> none , BatchNorm2d
- Convolutional layers -> 2, 4, 6
- Filters -> 16, 32, 64, 128
- Optimizers -> ADAM, SGD


The most important results:
- Batch normalization resulted in faster convergence
- Dropout of 0.2 gave the best balance between overfitting/slow convergence
- Models with more than 4 convolutional layers did not improve significantly
- The ADAM optimizer did way better than the SDG

- My best result was with a convolution with 32 filter + batchnorm = 88.5% accuracy

Reflection:
- Dropout helps to generalize the model by turning off random neurons, but it is recommended to stay below a dropout 0f 0.3
- Batch normalization stablise the activations and offers higher learning rates.
- Too deep networks gave diminishing returns
- MLFLOW comes in really handy when comparing the different runs of the model. And helps visualise hyperparamters.
Find the [instructions](./instructions.md)

[Go back to Homepage](../README.md
