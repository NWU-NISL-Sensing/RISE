# RISE: Robust Wireless Sensing using Probabilistic and Statistical Assessments

RISE is a generic framework for identifying and mitigating performance degradation issues when deploying a learning-based wireless sensing system in a changing environment. For more details, please refer to the [preprint version](https://arxiv.org/abs/2104.07460) of our paper, "RISE: Robust Wireless Sensing using Probabilistic and Statistical Assessments", which appeared in MobiCom 2021.

### Abstract

Wireless sensing builds upon machine learning classifiers shows encouraging results. However, adopting learning-based sensing as a large-scale solution remains challenging as experiences from deployments have shown the performance of a machine-learned model to suffer when there are changes in the environment, e.g., when furniture is moved or when other objects are added or removed from the environment. We present RISE, a novel solution for enhancing the robustness and performance of learning-based wireless sensing techniques against such changes during a deployment. 

RISE combines probability and statistical assessments together with anomaly detection to identify samples that are likely to be misclassified and uses feedback on these samples to update a deployed wireless sensing model. We validate RISE through extensive empirical benchmarks by considering 11 representative sensing methods covering a broad range of wireless sensing tasks. Our results show that RISE can identify 92.3% of mis-classifications on average. We showcase how RISE can be combined with incremental learning to help wireless sensing models retain their performance against dynamic changes in the operating environment to reduce the maintenance cost, paving the way for learning-based wireless sensing to become capable of supporting long-term monitoring in complex everyday environments.

### Citation

  @inproceedings{automated2021zhai,
  title={RISE: Robust Wireless Sensing using Probabilistic and Statistical Assessments},
  author={},
  booktitle={Mobicom},
  year={2021},
  organization={ACM}
}

### Resources

See [//RISE/Jupyter/](https://github.com/jiaojiao1234/RISE/tree/master/Jupyter) for the artifact interactive demo of the paper.

### Licenses

Released under the terms of the Apache-2.0 License. See [LICENSE](https://github.com/jiaojiao1234/RISE/blob/master/LICENSE) for details.
