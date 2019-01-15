# Kaggle-Quora-Insincere-Questions-Challenge
NLP using Convolutional Neural Networks(Kim Yoon 2015)

## Text Classification(NLP) using CNNs
A basic implementation of Kim Yoon's 2015 paper which reported improved results on a number of NLP tasks using Convolutional Neural Networks. The CNN layers/filters are trained on top of an embedding layer of pre-trained word vectors. *word2vec* embeddings are used in this model which were trained on a corpus of Google News in 2013. Word vectors are used to project words onto a lower dimensional vector space(300 in word2vec) and act as feature extractors that encode semantic features of words in their dimensions.
This can be loosely related to transfer learning in NLP. Pre-trained word vectors act as universal feature extractors for text much like early convolutional layers in image recognition which capture very basic features like edges,blobs etc.
## Model ##
The embedding layer is initialized by creating an embedding matrix. 3 convolutional layers with varying window sizes [3,4,5] are used to extract features followed by max pooling. The outputs of 3 max pooling layers are concatenated into a single tensor and then flattened for classification. A dropout with probability of 0.5 and ReLU activation is used as done in the paper.
The accuracy of the model is a misleading metric since the dataset is unbalanced. f1 score must be used to evaluate the model.
## Pre-processing ##
Pre-processing is carried out in such a way so as to increase the intersection between the dataset's vocabulary and the embeddings. Stop words are not removed while cleaning since the stop words(high frequency words) were already downsampled while training the word vectors and hence removal of stop words would make very small difference. The pre-processing ideas are largely drawn from other kaggle kernels(https://www.kaggle.com/christofhenkel).
## Data ##
* https://www.kaggle.com/c/quora-insincere-questions-classification/data
## RESOURCES
* https://arxiv.org/abs/1408.5882
* https://www.kaggle.com/christofhenkel
* http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
* http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/
* http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/
