# S3Former

**S3Former: Sequential, Structural, and Statistical Fusion for Continuous-Time Dynamic Graphs**

**NeurIPS 2026**

Ziyan Li, Xiangyu Hu, Tianqing Zhu, Wanlun Ma, Ke Qin

S3Former is a continuous-time dynamic graph learning framework for dynamic link prediction. It jointly models three complementary views of the causal interaction history:

- **Sequential Channel:** captures long-range temporal evolution and recent interaction dynamics.
- **Structural Channel:** constructs a causal shared pair subgraph and models explicit local connectivity with an edge-aware GNN.
- **Statistical Channel:** maintains causal node- and pair-level statistics, including activity, recency, interaction strength, temporal decay, and neighborhood overlap.

The three channels are integrated through a hierarchical context-aware collaborative fusion mechanism:

**Sequential / Structural / Statistical → NodeCCF → PairCCF → Link Prediction**

## Overview

Existing continuous-time dynamic graph models are effective at modeling temporal interaction histories, but serializing historical interactions may weaken explicit connectivity among candidate-centered historical context nodes.

S3Former addresses this limitation by jointly modeling temporal dynamics, causal local topology, and online statistical memory. NodeCCF adaptively combines endpoint-level information, while PairCCF performs candidate-specific relation correction using causal pair memory.

## Framework

<p align="center">
  <img src="assets/pipeline.png" width="95%">
</p>

S3Former consists of three complementary channels:

### Sequential Channel

The sequential channel models the historical interaction sequences of the source and destination nodes using long-range sequence encoding together with short-term temporal dynamics.

### Structural Channel

For each candidate interaction \((u,v,t)\), S3Former constructs a shared pair-centered subgraph using interactions strictly before \(t\). A lightweight edge-aware GraphSAGE module then encodes explicit historical connectivity among the selected context nodes.

### Statistical Channel

The statistical channel computes causal online statistics from historical information only, including node activity, pair interaction frequency, recency-aware strength, temporal decay, and neighborhood overlap.

### Context-aware Collaborative Fusion

S3Former uses two levels of adaptive fusion:

- **NodeCCF** integrates sequential, structural, and statistical evidence at the endpoint level.
- **PairCCF** performs candidate-specific relation correction using causal pair-level memory.

## Results

S3Former is evaluated on nine continuous-time dynamic graph benchmarks:

- Wikipedia
- Reddit
- MOOC
- LastFM
- Enron
- Social Evolution
- UCI
- Flights
- Contact

Under the standard transductive and inductive AP/AUC evaluation settings, S3Former achieves the strongest average rank across the four evaluation settings.

The paper additionally reports ranking-based evaluation on TGB-v2 datasets and robustness experiments under historical negative sampling.

## Requirements

The exact environment used for the released implementation will be provided here after verification.

```text
Python: <VERSION>
PyTorch: <VERSION>
CUDA: <VERSION>
GPU: NVIDIA RTX 4090 / RTX 4090 D
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Data

The experiments use publicly available continuous-time dynamic graph benchmarks from DyGLib, EdgeBank, and TGB.

Please refer to the corresponding benchmark repositories for the original datasets and preprocessing procedures.

Dataset preparation instructions for this repository will be provided in:

```text
data/README.md
```

## Training

Training commands will be added after the released implementation has been verified against the final NeurIPS 2026 version.

Example format:

```bash
python <TRAIN_SCRIPT> \
    --dataset wikipedia \
    <OTHER_ARGUMENTS>
```

## Evaluation

Evaluation commands will be provided for:

- transductive link prediction;
- inductive link prediction;
- TGB-v2 ranking evaluation;
- historical-negative robustness evaluation.

Example format:

```bash
python <EVAL_SCRIPT> \
    --dataset wikipedia \
    <OTHER_ARGUMENTS>
```

## Reproducing the Paper

We plan to provide scripts corresponding to the main experiments reported in the paper.

```text
scripts/
├── <main_benchmark_scripts>
├── <ablation_scripts>
└── <tgb_scripts>
```

The mapping between scripts and paper tables will be documented after verification.

## Repository Status

The repository is currently being prepared for the NeurIPS 2026 camera-ready release.

The implementation and reproduction commands will only be marked as final after they have been checked against the released paper configuration.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{li2026s3former,
  title={S3Former: Sequential, Structural, and Statistical Fusion for Continuous-Time Dynamic Graphs},
  author={Li, Ziyan and Hu, Xiangyu and Zhu, Tianqing and Ma, Wanlun and Qin, Ke},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

## Acknowledgments

This work was supported by the National Natural Science Foundation of China under Grant No. 62401123.

We thank Dr. Qiang Dong for helpful discussions on the theoretical analysis.

S3Former builds upon public continuous-time dynamic graph learning benchmarks and implementations, including DyGLib and related prior work.

## License

The license for this repository will be specified after confirming the appropriate release terms.
