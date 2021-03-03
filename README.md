# Iterative Eigenspace Update Algorithm for Image Recognition

Implementation of variant of Adaptive Eigenspace computaion algorithm proposed by Chandrashekharan et al. [[1]](#1) and consecutively using it for image recognition (Turk and Pentland)[[2]](#2).


## Dataset
- [Faces94 Dataset](https://cmp.felk.cvut.cz/~spacelib/faces/faces94.html) is a dataset consisting of 153 individuals. The subjects were sitting at approximately the same distance from the camera and were asked to speak while a sequence of twenty images was taken. The speech was used to introduce moderate and natural facial expression variation. For each subject there are about 20 images availble with facial variation.
- In perticular I use only Males for the Image recognition problem.
- Each image is RGB with shape (3,200,180) which is converted to grayscale image with shape (200,180).


## Usage
- Download the data from the source described above 
- Clone the repository to your local machine
<pre><code>
git clone https://github.com/AshishSinha5/eigen_update.git
</code></pre>
- Run main.py for feature extraction and testing (see main.py script for args help)
<pre><code>
python main.py -f "data/faces94/male/" -e 0.3 -n 3 -ntr 2
</code></pre>



### Train/Test Split
For each individual 1 training image is chosen for Eigenspace computation and a different random image is chosen for testing perposes.

## Methodology
- We first take the RGB images and convert it to grayscale and flatten it to form a vector of length 36000 (200 * 180).
- Following which we calculate the SVD of the image vectors iterative for each image and select the top k eigen-vectors which is determied by the parameter epsilon as discused by [[1]](#1).

![train_pipeline.png](https://github.com/AshishSinha5/eigen_update/blob/master/figures/train_pipeline.png)

- After extracting the eigen-vectors (eigenfaces) we need to able to represent our training Images as a linear combination of the these eigen-vectors. The coeffitients of linear combination are calculated as follows.
- 
![recontruction.png](https://github.com/AshishSinha5/eigen_update/blob/master/figures/reconstruction.png)

We can see how the weight vectos are created and reconstructed images are linear combinatiton of eigen-vectors

- Now that we have these weight vectors for each image, for any test image we again calculate the weights with similar methodology and find the most similar similar image in our dataset in terms of it's weight vector, this similarity measure can be any norm such L1, L2 or also Mahalanobis Distance.

![test_pipeline.png](https://github.com/AshishSinha5/eigen_update/blob/master/figures/test_pipeline.png)

## Inferences
- The accuracy for different levels of epsilon is shown below

![acc.png](https://github.com/AshishSinha5/eigen_update/blob/master/plots/epsilon_acc.png)

As we can see when we only take top 10% (eps = 0.1) of eigenvectors as our features in interative SVD update algorithm, the accuracy is low but it keeps on increasing as we increase the epsilon parameter.
The iterative SVD update algorithm is better since it lesser time complexity that what we would have in full SVD algorithm.

## References
<a id="1">[1]</a> S. Chandrasekaran, B.S. Manjunath, Y.F. Wang, J. Winkeler, H. Zhang, An Eigenspace Update Algorithm for Image Analysis, Graphical Models and Image Processing, Volume 59, Issue 5, 1997.

<a id="2">[2]</a> Matthew Turk and Alex Pentland, Eigenfaces for Recognition, 1991, MIT Press

