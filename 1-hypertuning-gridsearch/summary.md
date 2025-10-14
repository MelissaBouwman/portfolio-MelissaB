# Summary week 1

Hypothesis

I expected that increasing the number of units and/or epochs would improve accuracy. But there is a point where overfitting is going to occur. Therefore, i als expected that small learning rates yield more stable convergence than larger learning rate.

Experiments

I tested combinations of 
epochs: 5,7,10
units: 64, 128,256
Optimizer = ADAM
Learning rate: 1e-2, 1e-3, 1e-4

Results

Learning rate 1e-3 gave the best validation accuracy (89%)
The smaller learning rate diverged to early en the bigger one learned to slowly.
More units improved the training accuracy, but i also noticed that the validation accuracy did not improve. It seems like the model is slightly overfitting.

Discussion

I can conclude that more epochs improve the convergence, but there is a risk of overfitting because of the increased training time.
Also adding laters increased the porwer of the model but it is important to prevent the vanishing gradients.

Lessons Learned

A network (125-256 units) with a learning rate 1e-3 balances speed and good generalization.
Tensorbord comes in really handy for monitoring convergence trends.






Find the [notebook](./notebook.ipynb) and the [instructions](./instructions.md)

[Go back to Homepage](../README.md)
