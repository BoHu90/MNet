# MNet

Bo Hu, Shuaijian Wang, Leida Li, Jiaxu Leng, Yuzhe Yang, and Xinbo Gao, "Hierarchical discrepancy learning for image restoration quality assessment," Signal Processing, vol. 200, 2022, p. 108624.

# Abstract

With the unremitting efforts of the community, great progress has been made in image restoration (IR). However, how to objectively benchmark these algorithms is much less investigated and remains a challenging problem to date, which may hinder the rapid development of IR technologies. Most of the current IR quality metrics are scenario-special and cannot be generalized to diversified scenarios. Besides, they make predict solely on basis of restored versions, without using any information from the original degraded images. Therefore, they are blind to the relative perceptual discrepancy (RPD) between the degraded-restored image (DRI) pairs, which is much more useful when comparing the performance of IR algorithms. Inspired by this, we propose a hierarchical discrepancy learning (HIDI) model for benchmarking IR algorithms. To address the small sample problem, a quality evaluation database containing a variety of simulation distortions is built to pre-train a prior evaluator. Then, the DRI pairs are constructed and used to train the fully end-to-end HIDI model. The proposed HIDI model is extensively evaluated on several public databases and experimental results demonstrate the HIDI model has higher performance than the state-of-the-arts. Besides, compared with the competitors, the HIDI model shows the best generalization ability in cross-database experiments.

# Requirement

- numpy
- Pillow
- scipy
- torch
- torchvision
- tqdm

More information please check the requirements.txt.
