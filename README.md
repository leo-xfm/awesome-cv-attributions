# Awesome Computer Vision Models Attribution

**Continuously Update ...**



## Preliminaries

+ **Diffusion Models**: Denoising diffusion probabilistic models (DDPMs) (*Ho J, Jain A, Abbeel P., NeurIPS 2020*) [[paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)] [[homepage](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html)]



## Attribution Methods (Retraining-Based)

+ **[Empirical Influence]** What Neural Networks Memorize and Why: Discovering the Long Tail via Influence Estimation *(Feldman V, Zhang C., NeurlPS 2020)* [[paper](https://proceedings.neurips.cc/paper/2020/hash/1e14bfe2714193e7af5abc64ecbd6b46-Abstract.html?ref=the-batch-deeplearning-ai)]
+ **[Datamodel]** Datamodels: Predicting predictions from training data *(Ilyas A, Park S M, Engstrom L, et al. , ICML 2022)* [[paper](https://arxiv.org/abs/2202.00622)] [[code](https://github.com/MadryLab/datamodels-data)]
+ **[TRAK (ensemble)]** TRAK: Attributing Model Behavior at Scale  *(Park S M, Georgiev K, Ilyas A, et al., ICML 2023)* [[paper](https://arxiv.org/pdf/2303.14186)] [[code](https://github.com/MadryLab/trak)] [[homepage](https://trak.csail.mit.edu/)]
+ **[D-TRAK (ensemble)]** Intriguing Properties of Data Attribution on Diffusion Models *(Zheng X, Pang T, Du C, et al., ICLR 2024)* [[paper](https://arxiv.org/abs/2311.00500)] [[code](https://github.com/sail-sg/D-TRAK)] [[homepage](https://sail-sg.github.io/D-TRAK)] [[data](https://drive.google.com/drive/folders/1Ko1CI-nWo3NHWYpxfX2Un1t9UsuddVHX?usp=sharing)]



## Attribution Methods (Retraining-Free)

### 1. Similarity-based Techniques

+ **[Raw pixel and CLIP]** Learning transferable visual models from natural language supervision *(Radford A, Kim J W, Hallacy C, et al., ICML 2021)* [[paper](https://proceedings.mlr.press/v139/radford21a)]

### 2. Gradient-Based Techniques

#### a. Influence Functions (Hessian-based)

+ **[IF, Influence Function]** Understanding black-box predictions via influence functions (*Koh P W, Liang P., PLMR 2017 & ICML 2017 Best*) [[paper](https://proceedings.mlr.press/v70/koh17a/koh17a.pdf)] [[homepage](https://proceedings.mlr.press/v70/koh17a?ref=https://githubhelp.com)]
+ **[Gradient]** Input similarity from the neural network perspective *(Charpiat G, Girard N, Felardos L, et al, NeurlPS 2019)* [[paper](https://proceedings.neurips.cc/paper/2019/hash/c61f571dbd2fb949d3fe5ae1608dd48b-Abstract.html)]
+ **[Relatif]** Relatif: Identifying explanatory training samples via relative influence *(Barshan E, Brunet M E, Dziugaite G K., PMLR 2020)* [[paper](https://proceedings.mlr.press/v108/barshan20a.html)]
+ **[Renorm IF]** Identifying a Training-Set Attack's Target Using Renormalized Influence Estimation *(Hammoudeh Z, Lowd D, ACM SIGSAC 2022)* [[paper](https://dl.acm.org/doi/abs/10.1145/3548606.3559335)] [[code](https://github.com/ZaydH/target_identification)]
+ **[EK-FAC Influence]** Influence Functions for Scalable Data Attribution in Diffusion Models *(Mlodozeniec B, Eschenhagen R, Bae J, et al., 2025 ICLR)* [[paper](https://arxiv.org/abs/2410.13850)] 

#### b. Attribution Scores (Hessian-optimized)

+ **[TracIn]** Estimating Training Data Influence by Tracing Gradient Descent *(Pruthi G, Liu F, Kale S, et al., NeurIPS 2020)* [[paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/e6385d39ec9394f2f3a354d9d2b88eec-Paper.pdf)] [[code](https://github.com/frederick0329/TracIn)] [[homepage](https://proceedings.neurips.cc/paper/2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html)] 
+ **[GAS]** Identifying a training-set attack's target using renormalized influence estimation *(Hammoudeh Z, Lowd D, ACM SIGSAC 2022)* [[paper](https://dl.acm.org/doi/abs/10.1145/3548606.3559335)] [[code](https://github.com/ZaydH/target_identification)]
+ **[TRAK]** TRAK: Attributing Model Behavior at Scale  *(Park S M, Georgiev K, Ilyas A, et al., ICML 2023)* [[paper](https://arxiv.org/pdf/2303.14186)] [[code](https://github.com/MadryLab/trak)] [[homepage](https://trak.csail.mit.edu/)]
+ **[Journey TRAK]** The journey, not the destination: How data guides diffusion models *(Georgiev K, Vendrow J, Salman H, et al., ICML 2023)* [[paper](https://arxiv.org/abs/2312.06205)] [[code](https://github.com/MadryLab/journey-TRAK)] [[homepage](https://gradientscience.org/diffusion-trak/)]
+ **[D-TRAK]** Intriguing Properties of Data Attribution on Diffusion Models *(Zheng X, Pang T, Du C, et al., ICLR 2024)* [[paper](https://arxiv.org/abs/2311.00500)] [[code](https://github.com/sail-sg/D-TRAK)] [[homepage](https://sail-sg.github.io/D-TRAK)] [[data](https://drive.google.com/drive/folders/1Ko1CI-nWo3NHWYpxfX2Un1t9UsuddVHX?usp=sharing)]
+ **[D-ReTrac]** Data Attribution for Diffusion Models: Timestep-induced Bias in Influence Estimation *(Xie T, Li H, Bai A, et al., TMLR 2024)* [[paper](https://arxiv.org/abs/2401.09031)] [[code](https://github.com/txie1/diffusion-ReTrac)]
+ **[DAS]** Diffusion Attribution Score: Evaluating Training Data Influence in Diffusion Model *(Lin J, Tao L, Dong M, et al., ICLR 2025)* [[paper](https://arxiv.org/abs/2410.18639)] [[code](https://anonymous.4open.science/r/Diffusion-Attribution-Score-411F/README.md)]

### 3. Model-Based Techniques

+ **[AbC &  Custom Diffusion]** Evaluating Data Attribution for Text-to-Image Models *(Wang S Y, Efros A A, Zhu J Y, et al., ICCV 2023)* [[paper](https://arxiv.org/abs/2306.09345)] [[code](https://github.com/peterwang512/GenDataAttribution)] [[homepage](https://peterwang512.github.io/GenDataAttribution/)] [[data](https://github.com/peterwang512/GenDataAttribution#dataset)]
+ Data Attribution for Text-to-Image Models by Unlearning Synthesized Images *(Wang S Y, Hertzmann A, Efros A, et al., NeurIPS 2025)* [[paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/07fbde96bee50f4e09303fd4f877c2f3-Abstract-Conference.html)] [[code](https://github.com/PeterWang512/AttributeByUnlearning)] [[homepage](https://peterwang512.github.io/AttributeByUnlearning/)]



## Attribution Quantification & Evaluation

+ **[Encoded ensembles]** Training Data Attribution for Diffusion Models *(Dai Z, Gifford D K., arXiv 2023)* [[paper](https://arxiv.org/abs/2306.02174)] [[code](https://github.com/zheng-dai/GenEns)]
+ **[Linear Datamodeling Score]** TRAK: Attributing Model Behavior at Scale  (*Park S M, Georgiev K, Ilyas A, et al., ICML 2023*) [[paper](https://arxiv.org/pdf/2303.14186)] [[code](https://github.com/MadryLab/trak)] [[homepage](https://trak.csail.mit.edu/)]
