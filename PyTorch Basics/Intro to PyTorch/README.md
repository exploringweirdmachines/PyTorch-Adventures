## Intro to PyTorch: Exploring the Mechanics &nbsp; [<img src="../../src/visuals/x_logo.png" alt="drawing" style="width:25px;"/>](https://x.com/data_adventurer/status/1834073826612707543)&nbsp; [<img src="../../src/visuals/play_button.png" alt="drawing" style="width:40px;"/>](https://youtu.be/d86lJxKInYg?feature=shared) &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YQanR0ME7ThsU9YwLzXhGvYGOdH2ErSa?usp=sharing)
Deep Learning has revolutionized AI architectures, and it is vital to learn the tools to build these models! The reason we will be using PyTorch over the many other Deep Learning Frameworks out there is simply due to its 
current popularity! Obviously things change with time, but if you can build models with PyTorch the other frameworks
should come easy! All the following visuals were pulled from [here](https://www.assemblyai.com/blog/pytorch-vs-tensorflow-in-2023/)
in an analysis from Assembly AI.

![plot](https://cdn.prod.website-files.com/67a1e6de2f2eab2e125f8b9a/67be0ec7762fefb987e5ff56_Fraction-of-Papers-Using-PyTorch-vs.-TensorFlow.png)

The popularity of PyTorch in research has skyrocketed, mainly due to its Pythonic structure, relative ease of use,
and large opportunity for customization. More important is, many of the large Open Source efforts like the
HuggingFace Platform 🤗 have a larger focus towards building on top of PyTorch over Tensorflow.

![plot](https://cdn.prod.website-files.com/67a1e6de2f2eab2e125f8b9a/67be0ec8aef316bdeea12871_percentage_repo_2023.png)

Although many of the top models are offered in both PyTorch and Tensorflow variants, some models are **ONLY** available in 
PyTorch. 

If this is your first time tackling deep learning, this is a really exciting time! These topics are hard, but my goal
is to build you up step-by-step so that you can go from implementing simple Linear Models all the way to state-of-the-art 
Transformers in a sequential manner. We will cover the following things to get acquainted with the PyTorch system and review some basic ideas!

- PyTorch Fundamentals
    - Tensors
    - AutoGrad
- Optimization through Gradient Descent
- Linear Regression and Mean Squared Error Loss
- Logistic Regression and Binary Cross Entropy Loss
- MNIST Classification with Dense Neural Network
- Utilizing GPU to accelerate compute
