# Superposition -- Paper Replication
This repo is a replication of the paper 'Toy Models of Superposition', by Elhage et al. https://transformer-circuits.pub/2022/toy_model/index.html

It is work in progress; unfinished sections are on branches. Below you find short summaries of the finished (though unpolished) sections.


## Section summaries

### Section 2 -- Demonstrating Superposition
My results corresponding to Section 2 of the paper are presented in the notebook demo-superposition.ipynb and mostly agree with the findings outlined in the paper. We consider toy models with ReLU output layer. We see that whether they store features in superposition depends on the sparsity of the features in the training data. In the non-sparse regime the most important features have dedicated orthogonal directions and the remaining features are not represented. With increasing sparsity, more and more features are represented and stored in superposition, first in pairs and then in groups.

The code for training and visualization can be found in the respective files.

### Section 3 -- Superposition as Phase Change
A brief exposition of the results for this section is included in the notebook demo-superposition.ipynb. We observe a phase change as described in the paper: The optimal weight configuration (superposition versus no superposition) changes discontinuously with varying sparsity and relative importance. In difference to the paper, our training process (depending on initialization) sometimes runs into local minima. The phase diagrams that average over results of multiple models are therefore more noisy; if we pick the model with minimal loss instead, we see the phase change clearly. For more details, see the code file Section3main.py.

### Section 4 -- Geometry of Superposition
We include a brief exposition of the results for two subsections of this section in the demo notebook, namely the sections on uniform superposition and feature dimensionality. As described in the paper, we see that there are a few preferred geometric configurations to store features and these vary with sparsity. In particular, we observe digons (2 features stored as antipodal pair in 1 dimension) and pentagons (five features stored as pentagon in two dimensions). We do not observe tetrahedra, and the triangle configuration might require further training or closer analysis; or, most promising, an incentive to learn basis-aligned features (which is not incentivised by default in the dense regime). For the implementation details, compare Section4main.py and vizalization.py.