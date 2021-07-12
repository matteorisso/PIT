Copyright (C) 2021 Politecnico di Torino, Italy. SPDX-License-Identifier: Apache-2.0. See LICENSE file for details.

Authors: Matteo Risso, Alessio Burrello, Daniele Jahier Pagliari, Francesco Conti, Lorenzo Lamberti, Enrico Macii, Luca Benini, Massimo Poncino

![logo](Assets/logo.png)
# Pruning In Time (PIT): A Lightweight Network Architecture Optimizer for Temporal Convolutional Networks
[DAC 2021] Pruning In Time (PIT): A Lightweight Network Architecture Optimizer for Temporal Convolutional Networks

## Reference
If you use PIT in your experiments, please make sure to cite our paper:
```
@inproceedings{risso2021pit,
	author = {Risso, Matteo and Burrello, Alessio and Jahier Pagliari, Daniele and Conti, Francesco and Lamberti, Lorenzo and Macii, Enrico and Benini, Luca and Poncino, Massimo},
	title = {Pruning In Time (PIT): A Lightweight Network Architecture Optimizer for Temporal Convolutional Networks},
	year = {2021},
	publisher = {IEEE Press},
	booktitle = {Proceedings of the 58th ACM/EDAC/IEEE Design Automation Conference},
	series = {DAC '21}
}
```

## Abstract
Temporal Convolutional Networks (TCNs) are promising Deep Learning models for time-series processing tasks. One key feature of TCNs is time-dilated convolution, whose optimization requires extensive experimentation. We propose an automatic dilation optimizer, which tackles the problem as a weight pruning on the time-axis, and learns dilation factors together with weights, in a single training. Our method reduces the model size and inference latency on a real SoC hardware target by up to 7.4x and 3x, respectively with no accuracy drop compared to a network without dilation. It also yields a rich set of Pareto-optimal TCNs starting from a single model, outperforming hand-designed solutions in both size and accuracy.

## Requirements
- Python 3.6+
- Tensorflow 2.1.0
- Tensorflow-probability 0.9.0
- Scikit-learn 0.23.2
- Scikit-image 0.17.2
- Pandas 1.1.3

## Datasets
The current version support the following datasets:
- PPG_Dalia.
- Nottingham.
- JSB_Chorales.
- SeqMNIST (i.e., Sequential MNIST).
- PerMNIST (i.e., Permuted Sequential MNIST).

Further deitails about the pre-processing and data-loading phases of these datasets are provided under the **./Dataset** directory.

## How to run
Simply run:
```python
python pit.py <dataset> <reg_strength> <warmup_epochs>
```

## License
PIT is released under Apache 2.0, see the LICENSE file in the root of this repository for details.