---
layout: post
title:  "Data Preprocessing"
date:   2017-01-09 16:35:15 +0200
categories: main
---
<h1>Aside</h1>
Data Preprocessing play a very essential role in the area of [Knowledge Discovery][KDD-link].  Usually the data in our live world is not whiten yet. The images are sometimes very noised i.g. by taken under improperly illumination conditions or the image itself is not clean etc. 

<h1>Gaussian Filter</h1>
Gaussian filter enables us to remove the unnecessary information in images and retain the information that might important for representing the image. The basic idea is we just use some convolution operations with given kernel (ususally a matrix) to blur the small structure in images out, whereas the rough-textured blobs will be preserved. Usually the input image is a single channel gray image $$\mathbf{S}_i = (s_i(x, y))$$. The pixel points  $$s_o(x, y)$$  in output  $$\mathbf{S}_o$$ is given by a convolution operation $$\mathbf{K} = (h(u, v))$$ with its neighbor point $$s_i(x,y)$$. This operation is defined as follows: 

$$\begin{equation}
s_o(x, y) = \frac{1}{m^2} \sum_{u=0}^{m-1} \sum_{v=0}^{m-1} s_i(x + k - u, y + k - v) \cdot h(u, v)
\end{equation}$$

where $$m$$ the filter size of the filter $$\mathbf{K}$$  and $$ k = \frac{m-1}{2}$$. Mostly it's given by $$ m = 3, 5, 7, \dots$$. 
<div class="fig figcenter fighighlight">
  <img src="{{ site.github.url }}/assets/lena_gd.png" width="45%">
  <img src="{{ site.github.url }}/assets/lena_g.png" width="45%">
  <div class="figcaption"><b>Left: </b> original image <b>Right: </b> the image after Gaussian filter</div>
</div>

How much information will be blured depends closely on the size of filter. It's a hyperparameter.
{% highlight c++ %}
...
Mat tmp;
cv::GaussianBlur( img, tmp, cv::Size( 7, 7 ), 0 );
...
{% endhighlight %}

<h1>Digitalization</h1>
Some feature extraction needs binary image for extracting features such as histogram. So the original images should be digitalized, from which we got black and white points. Here we used [Otsu][otsu-link] method that clusters the pixels according to a threshold. The digitalized image looks like in figure below:
<div class="fig figcenter fighighlight">
  <img src="{{ site.github.url }}/assets/otsu.png" width="5%">
  <img src="{{ site.github.url }}/assets/otsu_o.png" width="5%">
  <div class="figcaption"><b>Left:</b>original image<b>Right:</b> the result after Otsu clustering</div>
</div>
To get this image, we can use this function given by OpenCV
{% highlight c++ %}
...
//thresholding to get a binary image
cv::threshold( tmp, tmp, 40, 255, THRESH_BINARY + CV_THRESH_OTSU );
...
{% endhighlight %}
<h1>Normalization</h1>
In our case the meters have various size. So the cropped images have also different size. The image ratio should be normalized without information lost, cause it also influence on the classification results. 

<h1>Principle Component Analysis</h1>
Principle Component Analysis [(PCA)][PCA-link] is a widely used dimension reduction method in data preprocessing or data clustering area. The basic idea behind this algorithm is eigenvalue decomposition. We always search a direction, when the original data onto this direction or direction of a vector the variance of the projected data is maximized. In other words, the data is approximated as the data lying in this direction or space. If we have an unit vector $u$ and a data point $$x$$, the length of the project of $$x$$ onto $$u$$ is $$x^T u$$. Also this projection onto $$u$$ is also the distance $$x^T u$$ from origin. Therefore we maximize the variance of projection by choosing unit-length $$u$$ as in following Eq.:

$$\begin{equation} 
	\begin{split}
	\frac{1}{m} \sum_{i=1}^{m}(x_i^{T} u)^2 &= \frac{1}{m} \sum_{i=1}^{m} u^T x_i x_i^T u \\
	&=u^T \left( \frac{1}{m} \sum_{i=1}^{m} x_ix_i^T \right) u,
	\end{split}
\end{equation} $$

the term in parentheses is actually the covariance  matrix of original data. After solving this optimization problem we got the principle eigenvalue of covariance matrix $$\Sigma = \frac{1}{m} \sum_{i=1}^{m} x_ix_i^T$$.  We can project our data onto this eigenvalue vector $$u_1, \dots, u_k, k < n$$, where $$k$$ and $$n$$ the dimension of eigenvector and original data. The projected data is calculated as follows:

$$\begin{equation}
y_i = \left[ \begin{array}{c}
 u_1^Tx_i \\
 u_2^Tx_i \\
 \vdots \\
 u_k^Tx_i
 \end{array} \right] \in \mathbb{R}^k.
\end{equation}$$

<div class="fig figcenter fighighlight">
  <img src="{{ site.github.url }}/assets/pca_o.png" width="50%">
  <img src="{{ site.github.url }}/assets/pca_r.png" width="50%">
  <img src="{{ site.github.url }}/assets/pca_e.png" width="50%">
</div>

The mosttop image is normalized by subtracting the mean image. The image after dimension reduction is shown in middle.The bottom image shows the 144 eigenvalues.



<h1>Whiten</h1>
The only difference between PCA and [Whitening] [whiten-link] is that the projected data is divided by the square root of eigenvalues.
$$\begin{equation}
y_{w} = \frac{y_i}{\sqrt{\lambda_i}}
\end{equation}$$

[KDD-link]:https://en.wikipedia.org/wiki/Data_mining
[otsu-link]:http://ijarcet.org/wp-content/uploads/IJARCET-VOL-2-ISSUE-2-387-389.pdf
[PCA-link]:https://en.wikipedia.org/wiki/Principal_component_analysis
[whiten-link]:http://ufldl.stanford.edu/tutorial/unsupervised/PCAWhitening/
