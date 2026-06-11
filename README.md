[![arXiv](https://img.shields.io/badge/arXiv-Preprint-b31b1b.svg)](https://arxiv.org/abs/2509.16625) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Self-Supervised Learning of Graph Representations for Network Intrusion Detection

This repository is based on the original implementation of **GraphIDS** presented in the paper *Self-Supervised Learning of Graph Representations for Network Intrusion Detection*. The goal of this project is to reproduce the results reported by the authors, evaluate the model on additional datasets.

<p align="center">
  <img src="figures/full_pipeline.png" alt="Graph representation learning process">
</p>

## Repository Structure

### `config_search_space/`

Contains the hyperparameter search space definitions used during tuning experiments.

- `tuning_space.yaml`: defines the range of values explored during hyperparameter optimization (e.g., learning rate, weight decay , mask ratio, window size, etc.).

### `configs/`

Contains dataset-specific configuration files.

New configuration files were added to support the additional datasets evaluated in this project, allowing experiments to be executed using the same training pipeline adopted by GraphIDS.

New configuration files:

```bash
ADFA-LD-GraphIDS.yaml
ADFA-LD-h2h.yaml
NF-ToN-IoT-v2.yaml
NF-ToN-IoT-v3.yaml
```

### `scripts/`

Contains dataset preprocessing utilities.

These scripts are responsible for converting raw datasets into the format required by GraphIDS, including feature extraction, graph construction, and dataset-specific transformations.

### `utils/parser.py`

Extended to support hyperparameter tuning through the following command-line arguments:

```bash
--tune
--tune_space
--tune_trials
--tune_seed
--tune_metric
--tune_num_epochs
--tune_patience
```

## Reproducibility

For detailed instructions on reproducing the experiments reported in the original paper, including environment setup, dataset preparation, and training commands, please refer to the official GraphIDS repository.

The modifications introduced in this repository are fully compatible with the original training pipeline. Additional experiments can be performed using the new configuration files available in configs/, the preprocessing scripts in scripts/, and the hyperparameter tuning functionality described above.

## Citation

```bibtex
@inproceedings{guerra2025selfsupervised,
  author = {Guerra, Lorenzo and Chapuis, Thomas and Duc, Guillaume and Mozharovskyi, Pavlo and Nguyen, Van-Tam},
  booktitle = {Advances in Neural Information Processing Systems},
  pages = {109471--109501},
  publisher = {Curran Associates, Inc.},
  title = {Self-Supervised Learning of Graph Representations for Network Intrusion Detection},
  url = {https://proceedings.neurips.cc/paper_files/paper/2025/file/9ddb13ae9150f99298065d889f951014-Paper-Conference.pdf},
  volume = {38},
  year = {2025}
}
```
