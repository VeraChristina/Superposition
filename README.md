# Superposition -- Paper Replication
This repo is a replication of the paper 'Toy Models of Superposition', by Nelhage et al. https://transformer-circuits.pub/2022/toy_model/index.html

It is work in progress; unfinished sections are on branches. Below you find short summaries of the finished (though unpolished) sections.


## Section summaries

### Section 2 -- Demonstrating Superposition
My results corresponding to Section 2 of the paper are presented in the notebook demo-superposition.ipynb and mostly agree with the findings outlined in the paper. We consider toy models with ReLU output layer. We see that whether they store features in superposition depends on the sparsity of the features in the training data. In the non-sparse regime the most important features have dedicated orthogonal directions and the remaining features are not represented. With increasing sparsity, more and more features are represented and stored in superposition, first in pairs and then in groups.

The code for training and visualization can be found in the respective files.

### Section 3 -- Superposition as Phase Change
A brief exposition of the results for this section is included in the notebook demo-superposition.ipynb. We observe a phase change as described in the paper: The optimal weight configuration (superposition versus no superposition) changes discontinuously with varying sparsity and relative importance. In difference to the paper, our training process (depending on initialization) sometimes runs into local minima. The phase diagrams that average over results of multiple models are therefore more noisy; if we pick the model with minimal loss instead, we see the phase change clearly. For more details, see the code file Section3-main.py.
