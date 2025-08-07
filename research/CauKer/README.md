# CauKer: Classification Time Series Foundation Models Can Be Pretrained on Synthetic Data Only

CauKer is a new synthetic data generation framework for pretraining classification Time Series Foundation Models (TSFMs) **without using any real data**.

🚀 **Best Time Series Paper and Spotlight Presentation @ ICML 2025 FMSD Workshop**  
📅 July 18, 2025 | 📍 Vancouver

## Overview

Recent work in TSFMs has focused on training with large-scale real-world corpora, which are expensive and hard to collect. **CauKer** tackles this challenge by generating synthetic time series data that are both:

- **Temporally realistic** (trends, seasonality, non-stationarity)  
- **Causally structured** (DAG-based propagation of Gaussian Process root nodes)

Our method combines:
- Gaussian Process Kernel Composition
- Structural Causal Graphs
- Rich activation and mean function libraries

With CauKer, we show that state-of-the-art classification models like **Mantis** and **MOMENT** can be pretrained entirely on synthetic data and still outperform or match real-data baselines in zero-shot classification tasks.

A more efficient version and training code will be released soon.

---

## Quick Start

> **New:** We have also uploaded **CauKer_V2.py**, a faster and more secure version of the framework.

You can try CauKer directly in your browser via Google Colab:

👉 [**Open Tutorial in Colab**](https://colab.research.google.com/drive/1hvVsWMP4g3pv9bqFRsgBolVMFBNF4tQk?usp=sharing)

```bash
# 1. Clone the repo
git clone https://github.com/ShifengXIE/CauKer.git
cd CauKer

# 2. Install dependencies


# 3. Generate 200,000 synthetic time series (default: 512-length, 4-dimensional)
python CauKer.py -N 200000 -L 512 -F 4 -P 6 -M 18 -O CauKer200K.arrow
```

---

## Example Use Case

Once generated, the synthetic `.arrow` dataset can be used to pretrain your own TSFM (e.g., Mantis or MOMENT) and then evaluated in a zero-shot setting on UCR, UEA benchmarks.

---

## Citation

```bibtex
@inproceedings{cauker2025,
  title={CauKer: Classification Time Series Foundation Models Can Be Pretrained on Synthetic Data Only},
  author={Shifeng Xie, Vasilii Feofanov, Marius Alonso, Ambroise Odonnat, Jianfeng Zhang, Ievgen Redko},
  booktitle={ICML Workshop on Foundation Models for Structured Data (FMSD)},
  year={2025}
}
```

---

## ❤️ Acknowledgements

This work is the result of a great collaboration—thanks to all my amazing co-authors: Vasilii Feofanov, Marius Alonso, Ambroise Odonnat, Jianfeng Zhang and Ievgen Redko, for their guidance and support throughout this project. Thanks MOAKHER Yessin for propose some ideas to be faster and safe.

If you have questions, collaboration ideas, or just want to discuss TSFMs, feel free to reach out:

📬 **shifeng.xie@telecom-paris.fr** **ievgen.redko@huawei.com>**
