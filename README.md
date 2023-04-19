# Superposition -- Paper Replication
This repo is a replication of the paper 'Toy Models of Superposition', by Nelhage et al. https://transformer-circuits.pub/2022/toy_model/index.html

It is work in progress; unfinished sections are on branches. Below you find short summaries of the finished sections.


## Section summaries

### Section 2 -- Demonstrating Superposition
My results corresponding to Section 2 of the paper are presented in the notebook demo-superposition.ipynb and mostly agree with the findings outlined in the paper. We consider toy models with ReLU output layer. We see that whether they store features in superposition depends on the sparsity of the features in the training data. In the non-sparse regime the most important features have dedicated orthogonal directions and the remaining features are not represented. With increasing sparsity, more and more features are represented and stored in superposition, first in pairs and then in groups.

The code for training and visualization can be found in the respective files.
