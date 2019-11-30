## Introductory

* [The Variational Auto-encoder](https://ermongroup.github.io/cs228-notes/extras/vae/)
* [VAE Explained](http://anotherdatum.com/vae.html) - Yoel Zeldes
* [VAE Explained in detail](http://anotherdatum.com/vae2.html) - Yoel Zeldes - this and the previous one are very good introductions and starting points to read
* [VAE Intuition and Implementation](https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/)
* [From Auto Encoder to beta-VAE](https://lilianweng.github.io/lil-log/2018/08/12/from-autoencoder-to-beta-vae.html) - excellent explanation in the section on VAE especially the math part
* [A Beginner's Guide to Variational Methods: Mean-Field Approximation](https://blog.evjang.com/2016/08/variational-bayes.html) - an introductory post but contains a great explanation of *why we should use reversed KL as opposed to forward KL* for minimizing the Loss Function of a VAE
* [Variational Autoencoders](https://ryanloweift6266.wordpress.com/2016/02/28/variational-autoencoders/) - This blog post is a usual introduction to the subject. But the best part I liked is the section *Problem Setup*, which is very intuitive.
* [Notes on Variational Autoencoders](http://www.1-4-5.net/~dmm/ml/vae.pdf) - Very detailed introduction.

## Original

* [Autoencoding Variational Bayes](https://arxiv.org/abs/1312.6114) - Kingma and Welling
* [An Introduction to Variational Autoencoders](https://arxiv.org/abs/1906.02691v2) - Kingma and Welling : A detailed exposition, updated recently (July 2019, check the latest version on arxiv)
* [Variational Lossy Autoencoder](https://arxiv.org/abs/1611.02731) - Xi Chen, Diederik P. Kingma, Tim Salimans, Yan Duan, Prafulla Dhariwal, John Schulman, Ilya Sutskever, Pieter Abbeel
* [Lagging Inference Networks and Posterior Collapse in Variational Autoencoders](https://arxiv.org/abs/1901.05534) - He et. al. - VAEs are known to suffer from a thing called "posterior collapse" where the true posterior equals approximate posterior which equals the prior. In such situation representation learning stops. This paper is a great analysis of this collapse.
* [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) - Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu
* [Generating Diverse High-Fidelity Images with VQ-VAE-2](https://arxiv.org/abs/1906.00446) - Ali Razavi, Aaron van den Oord, Oriol Vinyals
* [Improving Variational Autoencoder with Deep Feature Consistent and Generative Adversarial Training](https://arxiv.org/abs/1906.01984) - Xianxu Hou, Ke Sun, Linlin Shen, Guoping Qiu
* [DIVA: Domain Invariant Variational Autoencoders](https://arxiv.org/abs/1905.10427) - Maximilian Ilse, Jakub M. Tomczak, Christos Louizos, Max Welling
* [InfoVAE: Information Maximizing Variational Autoencoders](https://arxiv.org/abs/1706.02262)

## Tutorials

* [Variational AutoEncoders and Extensions](http://dpkingma.com/wordpress/wp-content/uploads/2015/12/talk_nips_workshop_2015.pdf) - NIPS 2015 tutorial by Kingma and Welling
* [Tutorial on Variational Encoders](https://arxiv.org/abs/1606.05908) - Carl Doersch
* [A Tutorial on Variational AutoEncoders with implementation](https://tiao.io/post/tutorial-on-variational-autoencoders-with-a-concise-keras-implementation/) - Has a nice Keras implementation
* [What is a Variational AutoEncoder](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/) - Explains VAE both from an NN perspective and from a probabilistic perspective. Very good explanations.
* [An Introduction to Variational Autoencoders](https://arxiv.org/abs/1906.02691) - Kingma and Welling - a monograph from the creators of VAE. **A long one and possibly the most detailed exposition**.
* [Variational Inference for Machine Learning](https://www.shakirm.com/papers/VITutorial.pdf) - a tutorial by Shakir Mohamed


## Reparamaterization Trick for VAEs

* [Useful StackExchange thread](https://stats.stackexchange.com/questions/199605/how-does-the-reparameterization-trick-for-vaes-work-and-why-is-it-important)
* [The Reparameterization Trick](http://gregorygundersen.com/blog/2018/04/29/reparameterization/) - an excellent blog post that really explains why we need the trick and what it actually means to say that *we cannot backpropagate through a random node*

## Background Readings

* [Predicting prob distributions using NNs](https://engineering.taboola.com/predicting-probability-distributions/) - discusses an approach to modeling probability distributions using NNs. Uses the Mixed Density Model of Bishop.
* [Neural variational inference and learning in belief networks](https://www.cs.toronto.edu/~amnih/papers/nvil.pdf) -  Mnih and Gregor
* [Transformations and expectations of random variables](http://www.its.caltech.edu/~mshum/stats/lect2.pdf)
* [Log Derivative Trick](http://blog.shakirm.com/2015/11/machine-learning-trick-of-the-day-5-log-derivative-trick/) by Shakir - a very useful trick used in section 2.2 of the original VAE paper (Autoencoding Variational Bayes). This blog post has a great explanation of this trick along with other places where it is used.
* [Notes on Multivariate Gaussian](http://cs229.stanford.edu/section/gaussians.pdf)
* [Variational Bayesian Inference with Stochastic Search](https://arxiv.org/abs/1206.6430) - John Paisley (UC Berkeley), David Blei (Princeton University), Michael Jordan (UC Berkeley)
* [Advances in Variational Inference](https://arxiv.org/pdf/1711.05597.pdf) - Cheng Zhang et. al. - This is a very lucid account of advances in VI starting with mean field inference and why we need amortized inference. Got the reference from this Quora thread [What is amortized variational inference](https://www.quora.com/What-is-amortized-variational-inference), also a very good explanation

## Additional Readings (Insights)

* [What is wrong with VAEs](http://akosiorek.github.io/ml/2018/03/14/what_is_wrong_with_vaes.html) - This post shows that in VAEs tighter evidence lower bounds (ELBOs) can be detrimental to the process of learning an inference network by reducing the signal-to-noise ratio of the gradient estimator. There is an accompanying paper [Tighter Variational Bounds are Not necessarily better](https://arxiv.org/abs/1802.04537) on this as well.
* [Variational Autoencoders do not train complex generative models](http://dustintran.com/blog/variational-auto-encoders-do-not-train-complex-generative-models) - a good insightful post.
* [Tutorial: Categorical Variational Autoencoders using Gumbel-Softmax](https://blog.evjang.com/2016/11/tutorial-categorical-variational.html) - VAEs for categorical distribution - discrete latent variables. The usual VAE is for continuous latent variables and Gaussian distribution. The latter is used for regression, the former for classification.
* Interesting threads
  * [Variational autoencoder: Why reconstruction term is same to square loss?](https://stats.stackexchange.com/questions/347378/variational-autoencoder-why-reconstruction-term-is-same-to-square-loss)
  * [What are the pros and cons of Generative Adversarial Networks vs Variational Autoencoders?](https://www.quora.com/What-are-the-pros-and-cons-of-Generative-Adversarial-Networks-vs-Variational-Autoencoders)

## Variational Inference and Neural Networks

* [Variational Inference and Deep Learning - a New Synthesis](https://pure.uva.nl/ws/files/17891313/Thesis.pdf) - Thesis of D Kingma
* [Practical Variational Inference for Neural Networks](https://papers.nips.cc/paper/4329-practical-variational-inference-for-neural-networks.pdf) - by Alex Graves

## Implementation

### VAE Implementation

* [Implementing Variational Autoencoders in Keras: Beyond the Quickstart Tutorial](http://louistiao.me/posts/implementing-variational-autoencoders-in-keras-beyond-the-quickstart-tutorial/)
* [Comprehensive Introduction to Autoencoders](https://towardsdatascience.com/generating-images-with-autoencoders-77fd3a8dd368) - The best part of this article is the implementation. Uses CNN for the deep learning part.
* [From expectation maximization to stochastic variational inference](https://nbviewer.jupyter.org/github/krasserm/bayesian-machine-learning/blob/master/variational_autoencoder.ipynb)
* [Deep feature consistent variational auto-encoder](http://krasserm.github.io/2018/07/27/dfc-vae/)
* [Tensorflow Implementation of MMD Variational Autoencoder](https://github.com/ShengjiaZhao/MMD-Variational-Autoencoder)
* [Variational Autoencoders with Tensorflow Probability Layers](https://medium.com/tensorflow/variational-autoencoders-with-tensorflow-probability-layers-d06c658931b7)
  * [Implementation on GitHub](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/vae.py) 


### Anomaly Detection using VAE

* [Variational Autoencoders for Anomaly Detection](https://rstudio-pubs-static.s3.amazonaws.com/308801_ca2c3b7a649b4fd1838402ac0cb921e0.html#/)
* [Variational Autoencoder based Anomaly Detection based on Reconstruction Probability](http://dm.snu.ac.kr/static/docs/TR/SNUDM-TR-2015-03.pdf)
* [Unsupervised Anomaly Detection via Variational Auto-Encoder for Seasonal KPIs in Web Applications](https://arxiv.org/abs/1802.03903) - Haowen Xu et al
  * [Implementation on GitHub](https://github.com/haowen-xu/donut)
* [Anomaly Detection on Network Intrusion Data](https://github.com/skeydan/anomaly_detection_VAE)


## Probabilistic Models, Gradient Estimation and Inference

### Probabilistic Models and Deep Learning

* [Probabilistic Models with Deep Neural Networks](https://arxiv.org/abs/1908.03442) by Andrés R. Masegosa, Rafael Cabañas, Helge Langseth, Thomas D. Nielsen, Antonio Salmerón - Great summary of how DNNs got married into the field of probabilistic models

### Gradient Estimation

* [Monte Carlo Gradient Estimation in Machine Learning](https://arxiv.org/abs/1906.10652) by Shakir Mohamed, Mihaela Rosca, Michael Figurnov, Andriy Mnih - Great survey on all comprehensive approaches. [Implementation](https://github.com/deepmind/mc_gradients)
* [Machine Learning Trick of the Day (4): Reparameterisation Tricks: Pathwise Estimator](http://blog.shakirm.com/2015/10/machine-learning-trick-of-the-day-4-reparameterisation-tricks/)
* [Machine Learning Trick of the Day (5): Log Derivative Trick: ScoreFunction Estimator](http://blog.shakirm.com/2015/11/machine-learning-trick-of-the-day-5-log-derivative-trick/)
* [REINFORCE vs Reparameterization Trick](http://stillbreeze.github.io/REINFORCE-vs-Reparameterization-trick/)

### Variational Inference and Expectation Maximization

* [The Variational Approximation for Bayesian Inference](http://www.cs.uoi.gr/~arly/papers/SPM08.pdf) - by Dimitris G. Tzikas, Aristidis C. Likas, and Nikolaos P. Galatsanos. This paper connects Bayesian Inference, EM and Variational EM. The paper assumes a few derivations which are well explained in the following 2 blog posts, along with some examples.
  * [Expectation Maximization and Variational Inference](https://chrischoy.github.io/research/Expectation-Maximization-and-Variational-Inference/) - provides some of the derivations that Tzikas et. al. skip in their article.
  * [Expectation Maximization and Variational Inference (Part 2)](https://chrischoy.github.io/research/Expectation-Maximization-and-Variational-Inference-2/)
* [A View of the EM Algorithm that Justifies Incremental Parse and other Variants](http://www.cs.toronto.edu/~fritz/absps/emk.pdf) - a classic by Radford Neal and Hinton
* [The EM Algorithm](http://cs229.stanford.edu/notes/cs229-notes8.pdf) - Stanford CS229 Study Notes
* [Motivation of Expectation Maximization Algorithm](https://stats.stackexchange.com/questions/64962/motivation-of-expectation-maximization-algorithm)
* [From expectation maximization to stochastic variational inference](http://krasserm.github.io/2018/04/03/variational-inference/)
* [Expectation Maximization](https://zhiyzuo.github.io/EM/)
* [David Blei's notes on Variational Inference](https://www.cs.princeton.edu/courses/archive/fall11/cos597C/lectures/variational-inference-i.pdf)
* A few Quora threads
  * [When should I prefer variational inference over MCMC for Bayesian analysis?](https://www.quora.com/When-should-I-prefer-variational-inference-over-MCMC-for-Bayesian-analysis)
  * [What is amortized variational inference?](https://www.quora.com/What-is-amortized-variational-inference)
  * [Variational Inference: Why do we use KL divergence instead of Jensen–Shannon divergence?](https://www.quora.com/Variational-Inference-Why-do-we-use-KL-divergence-instead-of-Jensen%E2%80%93Shannon-divergence)