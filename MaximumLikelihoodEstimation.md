# Maximum Likelihood Estimation

> maximum likelihood estimation (MLE) is a method of estimating the parameters of a probability distribution by maximizing a likelihood function, so that under the assumed statistical model the observed data is most probable.

[Wikipedia](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)

We want to estimate a true distribution $p^{*}(x)$. We don't know $p^{*}(x)$ but we can sample from it. Let's have a set of candidate distributions $\mathcal{P}_{x}$ (the hypothesis space). Each $p \in \mathcal{P}_{x}$ is a distribution that defines a density over ${x}$. Since we can sample from $p^{*}(x)$, we denote uniform sampling from this finite dataset as $\widehat{p}(x)$.

*Goal of MLE is to find a $p \in \mathcal{P}_{x}$ that best approximates $\widehat{p}(x)$ as measured by KL Divergence.*

Kevin Murphy's book says:

> maximizing likelihood is equivalent to minimizing $D_{K L}\left[P\left(\cdot | \theta^{*}\right) \| P(. | \theta)\right]$, where $P\left(\cdot | \theta^{*}\right)$ is the true distribution and $P\left(\cdot | \theta\right)$ is our estimate ..

**Proof:** 

$D_{K L}\left[P\left(\cdot | \theta^{*}\right) \| P(. | \theta)\right]$ 

$= 
\mathbb{E}_{x \sim P\left(x | \theta^{*}\right)}\left[\log \frac{P\left(x | \theta^{*}\right)}{P(x | \theta)}\right]$

$=\mathbb{E}_{x \sim P\left(x | \theta^{*}\right)}\left[\log P\left(x | \theta^{*}\right)-\log P(x | \theta)\right]$

$=\mathbb{E}_{x \sim P\left(x | \theta^{*}\right)}\left[\log P\left(x | \theta^{*}\right)\right]-\mathbb{E}_{x \sim P\left(x | \theta^{*}\right)}[\log P(x | \theta)]$

In the above, the left term is the entropy of $P\left(x | \theta^{*}\right)$ but does not depend on the estimated parameter $\theta$ - hence we can ignore that.

Now suppose we sample ${N}$ of $x \sim P\left(x | \theta^{*}\right)$. Then the law of large numbers says as ${N}$ goes to infinity:

$-\frac{1}{N} \sum_{i}^{N} \log P\left(x_{i} | \theta\right)=-\mathbb{E}_{x \sim P\left(x | \theta^{*}\right)}[\log P(x | \theta)]$, which is the right term of the above KL divergence.

And notice that $-\frac{1}{N} \sum_{i}^{N} \log P\left(x_{i} | \theta\right)=\frac{1}{N} \mathrm{NLL}$

Hence mimimizing KL divergence is equivalent to maximizing log likelihood.

