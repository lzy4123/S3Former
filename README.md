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

