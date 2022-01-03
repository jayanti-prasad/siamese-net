# Image Recognition with Siamese Net 

## Background 

   In ordinary image Recoginitipn based on Convolutional Neural Network (CNN) we train a CNN model
   with multiple convolutional, pooling etc., layers  for a set of labeled (finite) training image 
   and then use the trained model to predict the label of an unlabeled images. This scheme does not 
   work when the test image does not have the label from the trained label and it is not easy
   to incorporate a new label (we will need to do the full training from the scratch).

   Siamese network is not trained to predict the label of an unlabeled image but to answer the question 
   how different a test image is from a set of other images (may or may not be from the training data).

   Siamese network is based on a distance or similarity metrics. Siamese network is trained by showing 
   it pairs of images which are similar and different (in some ways). If similar and dissimilar images are 
   shown by the labels '0' and '1' respectively than it is possible to use binary cross entropy as loss.

   Note that before we compute the similarity distance between a pair of images we must extract features from 
   images and vectrorize those. For this purpose, we can with use conventional CNN network or use any pre-trained
   model (transfer learning). Here we train the model from the scratch, including embedding.

   Note that in place of passing pairs or similar and dissimilar sets of images, we can also pass a set of
   triplets of images (training image, similar image & dissimilar images) and use a 'triplet loss function'
   as is done here. 


## Advantages

   Once the model is trained it can work with a new set of images also -  needs just one example for matching.


## Reference 
   - [Image similarity estimation using a Siamese Network with a triplet loss
](https://keras.io/examples/vision/siamese_network/)

   - [A friendly introduction to Siamese Networks](https://towardsdatascience.com/a-friendly-introduction-to-siamese-networks-85ab17522942)
 
   - [Siamese networks with Keras, TensorFlow, and Deep Learning](https://www.pyimagesearch.com/2020/11/30/siamese-networks-with-keras-tensorflow-and-deep-learning/) 



