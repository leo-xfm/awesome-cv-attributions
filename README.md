# Awesome Large Vision Models Attribution  [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

***Continuously Update ...***

## What is Data Attribution?

Consider a model prediction problem. We have a dataset $D=\{z_1, ..., z_n \}$ of $n$ training examples, each of which is an input-label pair $z_i = (x_i, y_i) \in Z $, where $x_i \in \mathcal{X}^{n\times k}$ is the input and $y_i \in \mathcal{Y}$ is the output. Let $\mathcal{L}(z,\theta)$ be the loss function, and assume that model parameters are trained by minimizing the empirical risk  $\hat{\theta}=\arg\min_{\theta\in\Theta}\frac{1}{n}\sum_{i=1}^{n}\mathcal{L}(z_{i},\theta) =\arg\min_{\theta\in\Theta} \mathcal{R}(\theta) $, where $\mathcal{R}(\theta) = \frac{1}{n} \sum_{i=1}^{n} \mathcal{L}(z_i, \theta)$.

Now we define a model output function $f(z,\theta)$, which maps an interest point $z$ and model parameters $\theta\in\Theta$ to output predictions. A data attribution method $\tau(z,D)$ assigns a score to each point $z$ based on the training dataset $D$, measuring how much the training dataset $D$ influences the prediction of the model output function $f(z,\hat{\theta})$. Normally, the higher the attribution score is, the more contribution it has done to the interest point.

## Preliminaries

#### Diffusion Models

+ Denoising diffusion probabilistic models (DDPMs) (*Ho J, Jain A, Abbeel P., NeurIPS 2020*) [[paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)] [[homepage](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html)]
+ High-Resolution Image Synthesis With Latent Diffusion Models (LDMs)  (*Rombach, Robin, et al, CVPR 2022*) [[paper](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper)]
+ Multi-concept customization of text-to-image diffusion (Custom Diffusion) (*Kumari, Nupur, et al., CVPR 2023*) [[paper](http://openaccess.thecvf.com/content/CVPR2023/html/Kumari_Multi-Concept_Customization_of_Text-to-Image_Diffusion_CVPR_2023_paper.html)]

#### Influence Functions

+ Residuals and influence in regression, Ling, R. F. (1984) [[book](https://www.tandfonline.com/doi/pdf/10.1080/00401706.1984.10487996)]
+ **[IF, Influence Function]** Understanding black-box predictions via influence functions (*Koh P W, Liang P., PLMR 2017 & ICML 2017 Best*) [[paper](https://proceedings.mlr.press/v70/koh17a/koh17a.pdf)] [[homepage](https://proceedings.mlr.press/v70/koh17a?ref=https://githubhelp.com)]



## Attribution in Machine Learning

### A Quick Overview

+ [**Tutorial**] Data Attribution at Scale (*Andrew Ilyas, Kristian Georgiev, et al.*) [[homepage](https://ml-data-tutorial.org/)] [[video](https://icml.cc/virtual/2024/tutorial/35228)]

  > Data attribution is the study of the relation between data and ML predictions. In downstream applications, data attribution methods can help interpret and compare models; curate datasets; and assess learning algorithm stability.
  >
  > This tutorial surveys the field of data attribution, with a focus on what we call “predictive data attribution.” We first motivate this notion within a broad, purpose-based taxonomy of data attribution. Next, we highlight how one can view predictive data attribution through the lens of a classic statistical problem that we call “weighted refitting." We discuss why classical methods for solving the weighted refitting problem struggle when directly applied to large-scale machine learning settings (and thus cannot directly solve problems in modern contexts). With these shortcomings in mind, we overview recent progress on performing predictive data attribution for modern ML models. Finally, we discussing applications—current and future—of data attribution.



### Similarity-based Techniques

+ **[Raw pixel and CLIP]** Learning transferable visual models from natural language supervision *(Radford A, Kim J W, Hallacy C, et al., ICML 2021)* [[paper](https://proceedings.mlr.press/v139/radford21a)]
+ **[Representation-Smi]** Evaluation of similarity-based explanations (*Hanawa, Kazuaki, et al.,2021 ICLR*) [[paper](https://arxiv.org/abs/2006.04528)]



### Removal-Based Techniques

+ **[LOO]** Leave-One-Out
+ **[Data Shapley]** Data shapley: Equitable valuation of data for machine learning (*Ghorbani, Amirata, and James Zou., 2019 ICML*) [[paper](https://proceedings.mlr.press/v97/ghorbani19c.html)]
+ **[Empirical Influence]** What Neural Networks Memorize and Why: Discovering the Long Tail via Influence Estimation *(Feldman V, Zhang C., NeurlPS 2020)* [[paper](https://proceedings.neurips.cc/paper/2020/hash/1e14bfe2714193e7af5abc64ecbd6b46-Abstract.html?ref=the-batch-deeplearning-ai)]
+ **[Datamodel]** Datamodels: Predicting predictions from training data *(Ilyas A, Park S M, Engstrom L, et al. , ICML 2022)* [[paper](https://arxiv.org/abs/2202.00622)] [[code](https://github.com/MadryLab/datamodels-data)]



### Gradient-Based Techniques

+ **[IF, Influence Function]** Understanding black-box predictions via influence functions (*Koh P W, Liang P., PLMR 2017 & ICML 2017 Best*) [[paper](https://proceedings.mlr.press/v70/koh17a/koh17a.pdf)] [[homepage](https://proceedings.mlr.press/v70/koh17a?ref=https://githubhelp.com)]
+ **[Gradient]** Input similarity from the neural network perspective *(Charpiat G, Girard N, Felardos L, et al, NeurlPS 2019)* [[paper](https://proceedings.neurips.cc/paper/2019/hash/c61f571dbd2fb949d3fe5ae1608dd48b-Abstract.html)]
+ **[Relatif]** Relatif: Identifying explanatory training samples via relative influence *(Barshan E, Brunet M E, Dziugaite G K., PMLR 2020)* [[paper](https://proceedings.mlr.press/v108/barshan20a.html)]
+ **[Renorm IF & GAS]** Identifying a Training-Set Attack's Target Using Renormalized Influence Estimation *(Hammoudeh Z, Lowd D, ACM SIGSAC 2022)* [[paper](https://dl.acm.org/doi/abs/10.1145/3548606.3559335)] [[code](https://github.com/ZaydH/target_identification)]

+ **[TracIn]** Estimating Training Data Influence by Tracing Gradient Descent *(Pruthi G, Liu F, Kale S, et al., NeurIPS 2020)* [[paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/e6385d39ec9394f2f3a354d9d2b88eec-Paper.pdf)] [[code](https://github.com/frederick0329/TracIn)] [[homepage](https://proceedings.neurips.cc/paper/2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html)] 
+ **[TRAK]** TRAK: Attributing Model Behavior at Scale  *(Park S M, Georgiev K, Ilyas A, et al., ICML 2023)* [[paper](https://arxiv.org/pdf/2303.14186)] [[code](https://github.com/MadryLab/trak)] [[homepage](https://trak.csail.mit.edu/)]



## Attribution in Large Vision Models

### Gradient-Based Techniques

+ **[Journey TRAK]** The journey, not the destination: How data guides diffusion models *(Georgiev K, Vendrow J, Salman H, et al., ICML 2023)* [[paper](https://arxiv.org/abs/2312.06205)] [[code](https://github.com/MadryLab/journey-TRAK)] [[homepage](https://gradientscience.org/diffusion-trak/)]
+ **[D-TRAK]** Intriguing Properties of Data Attribution on Diffusion Models *(Zheng X, Pang T, Du C, et al., ICLR 2024)* [[paper](https://arxiv.org/abs/2311.00500)] [[code](https://github.com/sail-sg/D-TRAK)] [[homepage](https://sail-sg.github.io/D-TRAK)] [[data](https://drive.google.com/drive/folders/1Ko1CI-nWo3NHWYpxfX2Un1t9UsuddVHX?usp=sharing)]
+ **[Diff-ReTrac]** Data Attribution for Diffusion Models: Timestep-induced Bias in Influence Estimation *(Xie T, Li H, Bai A, et al., TMLR 2024)* [[paper](https://arxiv.org/abs/2401.09031)] [[code](https://github.com/txie1/diffusion-ReTrac)]
+ **[DAS]** Diffusion Attribution Score: Evaluating Training Data Influence in Diffusion Model *(Lin J, Tao L, Dong M, et al., ICLR 2025)* [[paper](https://arxiv.org/abs/2410.18639)] [[code](https://anonymous.4open.science/r/Diffusion-Attribution-Score-411F/README.md)]
+ **[EK-FAC Influence]** Influence Functions for Scalable Data Attribution in Diffusion Models *(Mlodozeniec B, Eschenhagen R, Bae J, et al., 2025 ICLR)* [[paper](https://arxiv.org/abs/2410.13850)] 



### Model-Based Techniques

#### Custom-Diffusion

+ **[AbC &  Custom Diffusion]** Evaluating Data Attribution for Text-to-Image Models *(Wang S Y, Efros A A, Zhu J Y, et al., ICCV 2023)* [[paper](https://arxiv.org/abs/2306.09345)] [[code](https://github.com/peterwang512/GenDataAttribution)] [[homepage](https://peterwang512.github.io/GenDataAttribution/)] [[data](https://github.com/peterwang512/GenDataAttribution#dataset)]
+ **[MONTRAGE]** MONTRAGE: Monitoring Training for Attribution of Generative Diffusion Models (*Brokman, Jonathan, et al., 2024ECCV*) [[paper](https://link.springer.com/chapter/10.1007/978-3-031-73226-3_1)]

#### Unlearning

+ **[Unlearning]** Data Attribution for Text-to-Image Models by Unlearning Synthesized Images *(Wang S Y, Hertzmann A, Efros A, et al., NeurIPS 2025)* [[paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/07fbde96bee50f4e09303fd4f877c2f3-Abstract-Conference.html)] [[code](https://github.com/PeterWang512/AttributeByUnlearning)] [[homepage](https://peterwang512.github.io/AttributeByUnlearning/)]
