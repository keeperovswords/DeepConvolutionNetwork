---
layout: post
title:  "Deep Learning"
date:   2017-04-11 14:28:55 +0300
categories: main
---
<h1>Prologue</h1>
As we mentioned in previous sections, the pipeline of image recognition can be polished by reconsidering its workflow. Now it looks as shows in figure:
<div class="fig figcenter fighighlight">
  <img src="{{ site.github.url }}/assets/pipeline.png" width="55%">
</div>
After the image is preprocessed as needed, the features will be extracted. Then these features are used for model training. This is actually so called <strong>Shallow Learning</strong>, which focuses on the expert knowledge in image processing in our task. It's kind of limitation here, cause the learning results also intensively depends on the feature selection part. In contrast to shallow learning, 
[Deep Learning][deeplearing-link] is raw data-oriented representation learning approach. It's normally supplied by the data in the raw form. There inputs are passed through a parametrized computational graph that's much more representational. This graph usually has many layers and optimized by [gradient-based schema][opt-link]. During the training the important features will be activated by this computational graph without extra feature extraction process.
Now the most popular deep visual learning model is Convolution Neural Network. Let's throught it out briefly.


<h1>Convolution Neural Network</h1>
Comparing with densely connected feed forward neural network, convolution neural network [(CNN)][cnn-link]  needs much few parameter. At one side it use sparsely connections in neural layers, at another side the subpooling mechanisim provides a probability to share parameters between neurons. I'm not going to depict is deeply, cause there are lots of articals talk about theoretically. You'd better to inform them yourself.Brielfy to say, it works af follows.
<h2>Forward propagation</h2>
With a convolution filter with fixed size the input neurons are not fully connected with the inputs. From it the feature will be activated. These features will be subsampled in subsampling layers. There are two subsampling strategies: <strong>Mean-Pooling</strong>, which calculates the mean in a pooling filter size. <strong>Max-Pooling</strong> determines the maximum of the activated maps in a pooling filter size. After subpooling the results will be continuously convoluted as needs.
 The activated maps after $$n$$- convolution layers are connected densely with feed forward neural network as we used before. The class scores are calculated in the last dense layer by using [Entropy loss][opt-link]. 
 With convolution operation and subpooling the parameters share there weight in a filter size. It's called <strong>weight-sharing</strong>.

<h2>Backward Propagation</h2>
 The scores is back propagated from last layer to the first convolution layer. For learning such a network by means of training examples, it turns to learn a input-output pattern correspondingly. The information of training samples is usually not sufficient by itself to reconstruct the unknown input-output mapping uniquely. Therefore we have to face the <strong>Overfitting</strong> problem. To conquer this issue we may use the [regularization][opt-link] method. Normally it has the following definition:
$$ \begin{equation}
(Regularized\ cost\ function) = (Empirical\ cost\ function) + ( Regularization\ parameter) \times ( Regularizer),
 \end{equation}$$

In our case the empirical cost is the entropy cost we got and we use $$L2$$ regularization as regularizer. The regularization paremeter $$\lambda$$ is determined normally by validation design.


<h2>Implementation</h2>

After showing the necessary techniques in deep learning consicely, let's go back to our classification problem. Here I used one convolution layer and a mean-pooling layer. The sub pooled features will be fed to a dense neural layer. Its strcuture is very simple. For optimizing the computational graph I used the Stochastic Gradient Descent [SGD+Momentum][sgd-link].
Ususally the SGD works alone well, but at the saddle points the cost function nudges very hesitated. In order to make the learner lernens effectively, the Momentum is introduced. As in following figure depicts, the cost function has benn reduced ineffectively at the first 20 iteration in first epoch. After then it converges very smoothly.
<div class="fig figcenter fighighlight">
  <img src="{{ site.github.url }}/assets/cnn_loss.png" width="95%">
  <img src="{{ site.github.url }}/assets/cnn_feature.png" width="65%">
  <div class="figcaption"><b>Top: </b> The cost function of cnn <b>Bottom: </b> the activated mapping in the first convolution layer </div>
</div>
As the input singal got forward propagated, the features turn to be much form or object oritend. I'm not showing  here that this structur is the best design. But it works for some numbers even better than MLP + HoG we trained previously as shows in the following figure
<div class="fig figcenter fighighlight">
  <img src="{{ site.github.url }}/assets/cnn_mlp.png" width="70%">
</div>
This model is actually not deep enough and it can be dramatically improved by tweaking its structur, i.e. say using ReLU for activation or [Batch Normalization][bt-link] to normalize the inputs in each convolution layer.

<h1>Conclusion</h1>
In contrast to models we introduced in previous sections, deep learning models need less image processing knowledge. The model strucutre has the ability to detect the most important features in inputs  on its own.



[deeplearing-link]:http://www.deeplearningbook.org/
[opt-link]:https://github.com/keeperovswords/Optimization
[sgd-link]:http://ufldl.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent/
[cnn-link]:https://en.wikipedia.org/wiki/Convolutional_neural_network
[bt-link]:https://arxiv.org/abs/1502.03167
