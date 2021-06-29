# CW-SOBA
Contextual Word embeddings in Semi-automatic Ontology Building for Aspect based sentiment analysis

## Recommended software
- Anaconda 3: https://www.anaconda.com/products/individual
- Python 3: https://www.python.org/downloads/
- PyCharm: https://www.jetbrains.com/pycharm/
- Protege: https://protege.stanford.edu/
- Java Eclipse IDE: https://www.eclipse.org/downloads/packages/release/kepler/sr1/eclipse-ide-java-developers
- Notepad++: https://notepad-plus-plus.org/downloads/

### GPU acceleration
Follow these steps to set up CUDA: https://towardsdatascience.com/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781
- NVIDIA CUDA 10.0: https://developer.nvidia.com/cuda-10.0-download-archive
- CuDNN 7.6.5.32 (for CUDA 10.0): https://developer.nvidia.com/cudnn (membership required)
- NVIDA Nsight Visual Studio Edition: (https://developer.nvidia.com/nsight-visual-studio-edition)
- Visual Studio 2017: https://visualstudio.microsoft.com/vs/older-downloads/

### Google Colab
Google Colab is needed for creating word embeddings and fine-tuning language models. A local runtime is advised for creating word-embeddings, as Google's TPU's are slow. When you don't have a GPU on your device, use a hosted runtime with Google's GPU's (recommended for fine-tuning language models).

Now follow the virtual environment set-up instructions from https://github.com/stefanvanberkum/CD-ABSC#set-up-instructions. Make sure to use this repository to include in the virtual environment. It is advised to leave the `ontology` directory out of the virtual environment, as no `.ipynb` files are present. _Note: Python 3.5 is highly recommended for the virtual environment, as higher versions often don't support TensorFlow 1 anymore. Using more recent versions of Python will throw a lot of errors during the installation of the packages in requirements.txt_

## Ontology building
The ontology building code is based on https://github.com/MarkRademaker/DCWEB-SOBA. We added some new functionality and fixed some bugs. The ontology building code can be found in the `ontology` directory.

Set-up steps:
- Get the Yelp data set: https://www.yelp.com/dataset
- 

## Evaluation
The evaluation code is based on https://github.com/stefanvanberkum/CD-ABSC.

