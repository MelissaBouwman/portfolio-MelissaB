# Welcome to My Data Science Portfolio

This portfolio contains a summary of my work, including machine learning exercises, a hackathon project, and critical reflections.

This website is built automatically using GitHub Pages. The main page you're reading right now is generated from the `README.md` file.

## Table of Contents

1.  **[Hyperparameter gridsearch](./1-hypertuning-gridsearch/summary.md)**
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

2.  **[Hyperparameter mlflow](./2-hypertuning-mlflow/summary.md)**

Hypothesis: I expected that adding a dropout would reduce the overfitting problem, especially in bigger networks with more filters/dense units. Therefore, i also expected that batch normalization would stabilise the training process and would speed up the convergence. WHile using more convolational layers i expected better validation results, this because the model can learn hierarchical feautures.

Results:

I used the FASHION-MNIST dataset with a batchsize of 64. Mainly i experimented with the following hyperparameters:

Dropout -> 0.0 0.2 0.5
Normalization -> none , BatchNorm2d
Convolutional layers -> 2, 4, 6
Filters -> 16, 32, 64, 128
Optimizers -> ADAM, SGD
The most important results:

Batch normalization resulted in faster convergence

Dropout of 0.2 gave the best balance between overfitting/slow convergence

Models with more than 4 convolutional layers did not improve significantly

The ADAM optimizer did way better than the SDG

My best result was with a convolution with 32 filter + batchnorm = 88.5% accuracy

Reflection:

Dropout helps to generalize the model by turning off random neurons, but it is recommended to stay below a dropout 0f 0.3
Batch normalization stablise the activations and offers higher learning rates.
Too deep networks gave diminishing returns
MLFLOW comes in really handy when comparing the different runs of the model. And helps visualise hyperparamters.
3.  **[Hyperparameter rnn](./3-hypertuning-rnn/summary.md)**

* During this assignment, I worked with the gesture recognition dataset and carried out several experiments to improve the RNN model.

I started with the basic RNN model, but I quickly noticed that the performance remained very low. With an input of 3 features, a hidden size of 68 and two layers, I was only able to reach an accuracy of about 9 percent, which is barely better than random guessing. From this, I concluded that the model was too simple for the complexity of the dataset.

After that, I tested a GRU model with a hidden size of 64 and a dropout of 0.2. After 65 epochs, this model reached an accuracy of 97%. I used the early stopping kwargs, which turned out to be really useful. The GRU performed much better and learned faster than the basic RNN model.

I also tried an LSTM model, which I had high expectations for, but it was quite disappointing because the accuracy did not go higher than 56 percent. Since the LSTM has many more parameters, it is also more sensitive to overfitting.

Next, I tried adding a Conv1D layer before the GRU, with 3 channels, 64 filters, and a kernel size of 5. What I noticed was that the model reached over 95% accuracy much earlier in the training process. This means it learned a lot faster than the GRU without the Conv1D layer.

From all these experiments, it became clear that the GRU is the best choice for this dataset. Models that were too deep or had too large a hidden size actually performed worse. The LSTM was not very efficient, and the basic RNN was far too limited. The addition of a Conv1D layer before the GRU gave the biggest improvement overall.

4.  **[Hyperparameter ray](./4-hypertuning-ray/summary.md)**
    * ## CNN Optimization with Ray Tune

In this experiment I optimized a Convolutional Neural Network (CNN) for the gestures dataset. The goal was to investigate the relationship between learning rate and batch optimization,

## 1. Hypothesis
I hypothesize that models with Batch Normalization will remain stable and achieve high accuracy even at high learning rates (e.g., > 0.01), whereas models without Batch Normalization will fail (diverge) at those same high learning rates due to internal covariate shift.

## 2. Methodology
I used ray tune to perform a hyperparameter search.
Algorithm: random search combined with grid search for batch norm.
Samples: 20 runs.
Search Space:
    * `Batch Normalization`: [True, False] (Grid Search)
    * `Learning Rate`: 1e-4 to 1e-1 (LogUniform)

## 3. Results
The visualization below shows the Validation Accuracy (Y-axis) against the Learning Rate (X-axis).
<img width="737" height="425" alt="image" src="https://github.com/user-attachments/assets/0b948b36-15f9-447b-a55c-1ed85f7d6281" />



* At lower learning rates (left side), both models perform well.
* As the learning rate increases beyond $10^{-2}$, the models without batch norm (blue circles) crash significantly, dropping to ~10% accuracy (random guessing).
* The models with batch norm** (green crosses) maintain high accuracy (~80%) even at the highest learning rates tested.

## 4. Conclusion
The experiment confirms my hypothesis. The visual clearly demonstrates that batch normalization acts as a stabilizer. It allows the neural network to be trained with much higher learning rates without diverging.

For future CNN architectures, I recommend always including batch normalization layers. They make the training process more robust and less sensitive to the specific choice of the learning rate.

5.  **[Hackathon Model: Project Cuddlefish](./6-hackathon/project_cuddlefish.md)**

### AI Challenge - Kadaster

The Kadaster registers over 100 types of legal events (rechtsfeiten), such as mortgages, seizures, or sales. However, recognizing these automatically is a major challenge because notaries use unstructured text without a fixed format. While standard models successfully identify common events, they fail on the Long Tail rare legal facts that occur infrequently (e.g., fewer than 20 times), making them impossible to learn via traditional training.

The Kadaster dataset suffers from a "Long Tail" distribution. While standard models (Regex/Neural) handle common "rechtsfeiten" well, they fail on rare legal facts where training data is scarce (fewer than 20 examples).

### Solution
To solve this, Christel and I developed a Zero-Shot Learning approach specifically for these rare cases. instead of training on examples, we used Large Language Models (LLMs) to recognize facts based on their legal descriptions.

### Methodology & Achievements

We benchmarked 5 different models and identified the top two performers based on context handling and accuracy:

- Qwen/Qwen3-30B-A3B-Thinking (262k context window)

- OpenAI/GPT-OSS-120B (131k context window)
  

- Advanced prompt engineering: we significantly improved performance by refining the prompt structure:

- Context injection: enriched the prompt with the rechtsfeiten definitions and added specific rules (e.g., synonyms).

- Noise reduction: instructed the model to ignore unnecessary historical information within the deeds.

- Hallucination prevention: built in an "escape mechanism" to ensure the model does not invent facts if the confidence is low.

- Configuration: we configured the system to target the long tail (threshold < 20 occurrences) with a context limit of 30,000 tokens to balance performance and speed.

### Results 
We successfully established a workflow that allows the system to recognize rare legal facts that the main model misses, with the Qwen and OpenAI models achieving comparable F1 scores (Micro F1 ~0.66).

6.  **[deployment](./5-deployment/summary.md)**

# AI Dog name generator
With the Straattaal set I created my own dog name generator. This AI model will come up with several new dog names generated based on a list with the most common dog names. 

<img width="1382" height="590" alt="image" src="https://github.com/user-attachments/assets/7ec1fa05-8f07-4420-94d3-a57253de59c5" />

<img width="1319" height="684" alt="image" src="https://github.com/user-attachments/assets/0815fb14-fe3e-4231-8ec3-04b670fd9001" />

It was a really nice project and i learned a lot, maybe in the future i will use it to name my dog(s) :P

the link of my repo: https://github.com/MelissaBouwman/MADS-deployment
7 .  **[Ethical Reflection: The Cuddlefish Dilemma](./7-ethics/summary.md)**
    * In this repo is a PDF file available for my ethics portfolio


## How to setup your own portfolio
- fork this repo
- go to the **settings** tab on top of your github page
- first, enable issues by scrolling down, and under "features" check the "issues" checkbox. 
- Then, on the left-hand menu click on the **Pages** tab
- under build and deployment, select `deploy from a branch`
- select main/root and save

After a few minutes you will see the site is published.

