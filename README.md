# Efficient differentially private kernel support vector classifier for multi-class classification},
Python implementation for 'Efficient differentially private kernel support vector classifier for multi-class classification},' published in Information Sciences Volume 619, January 2023, Pages 889-907.
Please refer to https://www.sciencedirect.com/science/article/pii/S0020025522011951.

### How to use

1. Run SVDD_DP.py file for default settings
   : You can change parameters in main function such as C (for SVDD), gamma (for RBF kernel), lr (for gradients when finding EPs), n_iter (for gradients when finding EPs), round_sep (for hypercube), averaged (for averaging n runs).
2. Specific Python implementataions for Multi-Basin Support-Based Clustering are in SVC.py.


### Requirement

scipy
sklearn
numpy
cvxopt



### Citation
```bibtex
@article{park2023efficient,
  title={Efficient differentially private kernel support vector classifier for multi-class classification},
  author={Park, Jinseong and Choi, Yujin and Byun, Junyoung and Lee, Jaewook and Park, Saerom},
  journal={Information Sciences},
  volume={619},
  pages={889--907},
  year={2023},
  publisher={Elsevier}
}
```
