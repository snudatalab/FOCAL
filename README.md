# FOCAL

This repository is the official implementation of 
"Fast and Accurate Online Coupled Matrix-Tensor Factorization via Frequency Regularization" (KDD 2026).

## Abstract

How can we efficiently and accurately factorize multi-source data in dynamic and real-time environments?
Coupled matrix-tensor factorization (CMTF) is a powerful tool for such tasks, but existing methods often struggle with scalability, particularly when dealing with continuously streaming data.
Traditional CMTF approaches, while effective at capturing complex relationships, suffer from computational inefficiencies and the need for retraining as new data arrive.
Moreover, many techniques fail to properly incorporate the inherent temporal characteristics of the data, which could significantly enhance both accuracy and convergence speed.

In this paper, we propose FOCAL (Frequency-regularized Online Coupled Approximation for Low-rank factorization), an efficient CP decomposition method designed to enhance online coupled matrix-tensor factorization.
By effectively distinguishing between old and new data, FOCAL optimizes computational efficiency, reducing redundant computations and eliminating the need for full retraining in streaming settings.
Furthermore, FOCAL integrates frequency regularization into an online CMTF framework, which mitigates overfitting and improves accuracy.
Through extensive experiments, we demonstrate that FOCAL outperforms existing state-of-the-art methods in terms of both speed and accuracy.
We also present results on anomaly detection using real-world data, showcasing FOCAL's effectiveness in identifying irregular patterns. 

## Prerequisites
Our code requires Tensor Toolbox (available at https://gitlab.com/tensors/tensor_toolbox).

## Datasets
The datasets are available at [ML-100K](https://grouplens.org/datasets/movielens/100k/), [Amazon](https://www.kaggle.com/datasets/deovcs/amazon-dataset), [Stock-US](https://drive.google.com/open?id=1w-eSA_BtjSlKu1pKN7RdMS8YmW5JgHGz&usp=drive_copy), [Stock-JPN](https://drive.google.com/open?id=1ZxCckBN7XLt6ZUXaPsz4n-NrR9E55sDB&usp=drive_copy), and [Stock-CHN](https://drive.google.com/open?id=1o5kCQ-0NtgVDfhe-VszdNowKoZ-u8Oz0&usp=drive_copy).

| Dataset      | Stream Type     | Tensor Size               | Matrix Size         |
|-------------|---------------|--------------------------|---------------------|
| **ML-100K**  | Single-Tensor  | $1682 \times 943 \times 200$   | $1682 \times 19$   |
| **Amazon**   | Single-Tensor  | $901 \times 1720 \times 200$  | $901 \times 170$   |
| **Stock-US** | Matrix-Tensor  | $500 \times 88 \times 3651$  | $3651 \times 10$   |
| **Stock-JPN** | Matrix-Tensor  | $300 \times 88 \times 3856$       | $3856 \times 10$       |
| **Stock-CHN**  | Matrix-Tensor  | $1471 \times 88 \times 1170$       | $1170 \times 10$       |


