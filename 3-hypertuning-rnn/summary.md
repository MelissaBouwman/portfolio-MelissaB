# Summary week 3
During this assignment, I worked with the gesture recognition dataset and carried out several experiments to improve the RNN model.

I started with the basic RNN model, but I quickly noticed that the performance remained very low. With an input of 3 features, a hidden size of 68 and two layers, I was only able to reach an accuracy of about 9 percent, which is barely better than random guessing. From this, I concluded that the model was too simple for the complexity of the dataset.

After that, I tested a GRU model with a hidden size of 64 and a dropout of 0.2. After 65 epochs, this model reached an accuracy of 97%. I used the early stopping kwargs, which turned out to be really useful. The GRU performed much better and learned faster than the basic RNN model.

I also tried an LSTM model, which I had high expectations for, but it was quite disappointing because the accuracy did not go higher than 56 percent. Since the LSTM has many more parameters, it is also more sensitive to overfitting.

Next, I tried adding a Conv1D layer before the GRU, with 3 channels, 64 filters, and a kernel size of 5. What I noticed was that the model reached over 95% accuracy much earlier in the training process. This means it learned a lot faster than the GRU without the Conv1D layer.

From all these experiments, it became clear that the GRU is the best choice for this dataset. Models that were too deep or had too large a hidden size actually performed worse. The LSTM was not very efficient, and the basic RNN was far too limited. The addition of a Conv1D layer before the GRU gave the biggest improvement overall.

Find the [notebook](./notebook.ipynb) and the [instructions](./instructions.md)

[Go back to Homepage](../README.md)
