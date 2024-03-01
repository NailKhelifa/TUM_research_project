<h1 align="center"> Technical University of Munich Research Project </h1> <br>
<p align="center">
  <a href="https://github.com/NailKhelifa/TUM_research_project">
    <img src="https://learningabroad.utoronto.ca/wp-content/uploads/TUM_logo.png" alt="Logo" width="400" height="400">
  </a>
</p>

<p align="center">
  <strong>Portfolio Optimization Using Clustering - Mixture Model.</strong>
</p>

<p align="center">
  Naïl Khelifa
  <br />
  <br />

  <a href="https://www.linkedin.com/in/naïl-khelifa-581665220/">
    <img src="https://img.shields.io/badge/LinkedIn-blue?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn Badge"/>
  </a>
  <a href="khelifa.nail@gmail.com">
    <img src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="gmail Badge"/>
  </a>
</p>



<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
## Table of Contents

- [Introduction](#introduction)
- [Assumption](#assumption)
- [Authors](#authors)
- [Licence](#license)
- [Acknowledgments](#acknowledgments)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Introduction

In Markowitz's Mean Variance Theory, investors focus on the mean and variance of portfolio returns. Portfolio optimization, known as **Mean-Variance Optimization (MVO)**, seeks to maximize returns while limiting variance. The optimization problem is expressed as:

```math
\min ~ \frac{1}{2}<\mathbf{\omega}, \Sigma \mathbf{\omega}>
\text{ subject to }
\quad \rho(\mathbf{\omega}) = \rho_0
\quad \sum_{i=1}^N \omega_i = 1
```

Here, $\Sigma$ is the covariance matrix, $\rho(\mathbf{\omega})$ is the reward for allocation $\mathbf{\omega}$, and $\rho_0$ is a fixed reward level.

For MVO, estimating returns and the covariance matrix $\Sigma$ is crucial. However, in a large asset universe (e.g., 700 stocks), inverting $\Sigma$ can be problematic due to estimation complexities. Estimating $\hat{\Sigma}$ may result in a non-invertible matrix, and even if $\Sigma$ is invertible, it is often ill-conditioned in a large asset universe.

To address this, traditional approaches modify $\hat{\Sigma}$ numerically or shrink it towards an invertible target matrix. However, these methods have drawbacks. Instead, we tackle these challenges using a different approach: clustering.

Our aim is to identify groups of similar returns through clustering, grouping assets for better diversification. We divide the assets into fewer subgroups (clusters) and allocate weights uniformly within each group based on clustering algorithms using return and fundamental data. Finally, classical MVO is performed on the new asset groups.

## Assumption

We consider $d, n \in \mathbb{N}$ and a set of observations $\chi = \{\mathbf{r}^{(1)}, ..., \mathbf{r}^{(n)}\}$ with $\mathbf{r}^{(i)} \in \mathbb{R}^d$ for each $i \in$ { $1$, ..., $n$ }, where $n$ represents the number of assets, $d$ the number of days, and $r^{(i)}_j$ the return of the $i$-th asset on day $j$ for $i \in$ { $1$, ..., $n$ } and $j \in$ { $1$, ..., $d$ }. 

In this approach, more than assuming that each new asset is a cluster, we assume that it is a probability distribution whose parameters we have to estimate. Let $K \in \mathbb{N}$ be the number of clusters (or components). That means we want to build a smaller portfolio composed of $K$ new assets. It stands to reason to use, in this case, Gaussian components, that is, to suppose that each cluster is a probability distribution that follows a $d$-multivariate Gaussian distribution. We denote $\mathcal{N}_d(\mu_1, \Sigma_1), ..., \mathcal{N}_d(\mu_K, \Sigma_K)$ these distributions with $\mu_k \in \mathbb{R}^d$ and $\Sigma_k \in \mathbb{R}^d$ for each $k \in$ { $1$, ..., $K$ }.

In our model, we do not consider that all data are generated from a single multidimensional law, but rather that they are taken out of a mixture of them. Therefore, we introduce the mixing probabilities $\pi_1, ..., \pi_K$ (that verify $\sum_k \pi_k = 1$ and $\pi_k > 0$ for each $k \in$ { $1$, ..., $K$ }) of each one of the components. In other words, $\pi_k$ corresponds to the probability of drawing the observations from the $k$-th component, i.e., from $\mathcal{N}_d(\mu_k, \Sigma_k)$.

After detailing our assumption that the data are distributed according to a Gaussian Mixture Model (GMM), there remains the problem of estimating the parameters. This is where a real problem arises.

In this setting, we denote $\Psi = \{\theta_1, ..., \theta_K, \pi_1, ..., \pi_K\}$ as the set of parameters to estimate for this model, where:

- For each $k \in$ { $1$, ..., $K$ }, $\theta_k = (\mu_k, \Sigma_k)$ is the set of parameters that determine the $k$-th Gaussian component $\mathcal{N}(\mu_k, \Sigma_k)$.
- $\pi_1, ..., \pi_K$ are the mixing probabilities corresponding to the GMM (thus $\sum_k \pi_k = 1$ and $\pi_k > 0$ for each $k \in$ { $1$, ..., $K$ }).

We consider a component $\mathcal{N}_d(\mu_k, \Sigma_k)$. In this case, the parameters to estimate are $\mu_k \in \mathbb{R}^d$ and $\Sigma_k \in \mathbb{R}^{d \times d}$. This means we have $d + \frac{d(d+1)}{2}$ parameters to estimate, in other words, a total of $K(d + \frac{d(d+1)}{2})$ parameters to estimate for the whole model.

For instance, if we consider a trading year of financial data, i.e., $d=250$ and $n \approx 500$, there are $K\frac{250^2}{2}$ parameters to estimate. The number of parameters to estimate evolves with $\frac{250^2}{2}$ times the number of clusters, but we have only $n \approx 500$ observations. This is precisely the problem.

This allows us to mention the more general problem, which is that of clustering in high dimension. Clustering in high-dimensional spaces is a difficult problem that is recurrent in many domains, for example, in image analysis. The difficulty is due to the fact that high-dimensional data usually live in different low-dimensional subspaces hidden in the original space. Methods based on the Gaussian Mixture Model (GMM) show a disappointing behavior when the size of the dataset is too small compared to the number of parameters to estimate, which is precisely our case.

## Our assumption

We thus have to reduce the number of parameters to estimate. To do so, we don't resort to any dimension reduction technique but we rather make a fairly basic assumption in financial mathematics.

In finance it is a common assumption to assume that returns of the **same** asset are uncorrelated and normally centred around a similar mean.

In our setting, this is tantamount to assume that the $K$ Gaussian regimes we introduced have the following form $\mathcal{N}_d(\mu_1, (\sigma_1)^2 \mathbb{I}_d)$, ..., $\mathcal{N}_d(\mu_K, (\sigma_K)^2 \mathbb{I}_d)$ with $\mu_k \in \mathbb{R}$, $\sigma_k \in \mathbb{R}^{+*}$ for each $k \in$ { $1$, ..., $K$ } and $\mathbb{I}_d
\in \mathbb{R}^{d \times d}$ denotes the identity matrix. 

In other words, this means that for each asset $r^{(i)}$, there exists a component in our mixture model whose label is denoted $k_{(i)}$ such that the daily returns we observe are centered around a similar mean value $\mu_{k_{i}}$ and deviate from this mean with a standard-deviation of $\sigma_{k_{i}}$.

In mathematical terms, this can be written as:

$$
\forall i = 1, ..., n, ~~~~ \exists k^{(i)} = \{1, ..., K \}, ~~~~ r^{(i)} \sim \mathcal{N}_d(\mu_k^{(i)}, (\sigma_k^{(i)})^2 \mathbb{I}_d)
$$

This precisely corresponds to the setting of a Gaussian mixture model. We still denote $\pi_1, ..., \pi_K$ the mixing probabilities of this mixture.

We resume this hypothesis by the following notations:

$$
\forall i = 1, ..., n, ~~~~ r^{i)} \sim \sum_{k=1}^K \pi_k \mathcal{N}_d (\mu_k^{(i)}, (\sigma_k^{(i)})^2 \mathbb{I}_d)
$$


Three remarks about this hypothesis:

- We have drastically reduced the number of parameters to estimate to $3K$ parameters.
- This easier setting should not hide a fundamental issue which is that of finding the right $K$.
- We have to estimate the mixing probabilities

Now we have simplified the estimation problem with our assumption. We remind that the idea of using Gaussian mixture models for clustering comes from a lack of control and visibility over the cluster. We believe this method will enable us to better understand the clustering stability issue.

## The estimation problem due to our assumption 

Given this hypothesis, we can adapt the global framework for the estimation problem. Considering the notations in $\textbf{\textit{Deng, H. and Han, J. (2014). Probabilistic models for clustering. Data Clustering, 1(0):61–82}}$ (which is part of the literature review I did last summer), we have in our case:

$$
\left\{
    \begin{array}{ll}
        \theta_k = \{\bm{\mu}_k, \Sigma_k\} = \{\mu_k \mathbf{e}, (\sigma_{k})^2 \mathbb{I}_d\} ~~ \forall k \in \{1, \hdots, K\}\\
        \\
        p(\mathbf{r}^{(i)} \lvert \theta_k) = \mathcal{N}_{d}(\mathbf{r}^{(i)} \lvert \mu_k \mathbf{e}, (\sigma_{k})^2 \mathbb{I}_d) \\
    \end{array}
\right.
$$

where \(\mathbf{e} = [1, \hdots, 1]^T \in \mathbb{R}^d\) and \(\mu_k \in \mathbb{R}\).

We also have:

\[
p(\mathbf{r}^{(i)} \lvert \Psi) = \sum_{k=1}^K \pi_k \mathcal{N}_{d}(\mathbf{r}^{(i)} \lvert \mu_k \mathbf{e}, (\sigma_{k})^2 \mathbb{I}_d)
\]

hence the following log-likelihood:

\[
\begin{equation}
    \mathcal{L}(\chi \lvert \Psi) = \sum_{i=1}^n \log \sum_{k=1}^K \pi_k \mathcal{N}_{d}(\mathbf{r}^{(i)} \lvert \mu_k \mathbf{e}, (\sigma_{k})^2 \mathbb{I}_d)
    \label{eq:likelihood}
\end{equation}
\]

Now we want to maximize this function. As said in the general setting, the optimal solution satisfies three equations that define the \(\gamma_k\)'s, \(\mu_k\)'s, and \(\pi_k\)'s. If we adapt them in our setting this gives:

\[
\begin{equation}
\label{eq:system}
\begin{cases}
\begin{aligned}
\mu_k &= \frac{\sum_{i=1}^{n} \gamma (z_{i,k})\mathbf{r}^{(i)}}{\sum_{n=1}^{n} \gamma (z_{i,k})} \\
\Sigma_k &= \frac{\sum_{i=1}^{n} \gamma (z_{i,k}) (\mathbf{r}^{(i)} - \mu_k)(\mathbf{r}^{(i)} - \mu_k)^T}{\sum_{i=1}^{n} \gamma (z_{i,k})} \\
\pi_k &= \frac{\sum_{i=1}^{n} \gamma (z_{i,k})}{n}
\end{aligned}
\end{cases}
\end{equation}
\]

where:

\[
\gamma(z_{i,k}) = \frac{\pi_k \mathcal{N}_{d}(\mathbf{r}^{(i)} \lvert \mu_k \mathbf{e}, (\sigma_k)^2\mathbb{I}_d)}{\sum_{j=1}^K \pi_j \mathcal{N}_{d}(\mathbf{r}^{(i)} \lvert \mu_j \mathbf{e}, (\sigma_j)^2\mathbb{I}_d)}
\]
The equations of \(\mu_k\), \(\Sigma_k\), and \(\pi_k\) are not the closed-form solution for the parameters of the mixture model. The reason is that these equations are intimately coupled with the equation of \(\gamma(z_{i,k})\). Therefore, maximizing the log likelihood function for a Gaussian mixture model turns out to be a very complex problem. A powerful method for finding maximum likelihood solutions for models with latent variables is called the **Expectation-Maximization algorithm**.

Here is the framework of the EM Algorithm:

```markdown
\begin{minipage}{\linewidth}
    **Framework of the EM algorithm for GMM:**
    1. **Initialization:** Initialize the means \(\mu_k^0\), covariances \(\Sigma_k^0\), and mixing probabilities \(\pi_k^0\).
    2. **Expectation-step:** Calculate the responsibilities \(\gamma (z_{i,k})\) using the current parameters based on the expression in ~\eqref{responsibilities}.
    3. **Maximization-step:** We use the expressions in ~\eqref{eq:system} to update the parameters using the current responsibilities. We *first* update the new means using, and *then* use these new values to calculate the covariances using, and *finally* re-estimate the mixing probabilities.
    4. Compute the log-likelihood using ~\eqref{eq:likelihood} and check for convergence of the algorithm. If the convergence criterion is not satisfied, then repeat step 2-4, otherwise, return the final parameters.
\end{minipage}


## Authors

  - Naïl Khelifa 

## License

Distributed under the [MIT License](LICENSE.md). See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Acknowledgments


[license-shield]: https://img.shields.io/badge/license-MIT-blue
[license-url]: https://github.com/NailKhelifa/PyFolioC/blob/main/LICENSE
