# Distance Learner: Incorporating Manifold Prior to Model Training

## About

The manifold hypothesis (real world data concentrates near low-dimensional manifolds) is suggested as the principle behind the effectiveness of machine learning algorithms in very high dimensional problems that are common in domains such as vision and speech. Multiple methods have been proposed to explicitly incorporate the manifold hypothesis as a prior in modern Deep Neural Networks (DNNs), with varying success. In this paper, we propose a new method, Distance Learner, to incorporate this prior for DNN-based classifiers. Distance Learner is trained to predict the distance of a point from the underlying manifold of each class, rather than the class label. For classification, Distance Learner then chooses the class corresponding to the closest predicted class manifold. Distance Learner can also identify points as being out of distribution (belonging to neither class), if the distance to the closest manifold is higher than a threshold. We evaluate our method on multiple synthetic datasets and show that Distance Learner learns much more meaningful classification boundaries compared to a standard classifier. We also evaluate our method on the task of adversarial robustness, and find that it not only outperforms standard classifier by a large margin, but also performs at par with classifiers trained via state-of-the-art adversarial training.

## Setup and Usage

### Dependencies

```
* Python 3.6+
* scikit-learn==0.22.1
* scipy==1.4.1
* numpy==1.18.1
* torch==1.11.0
* torchvision==0.12.0
* faiss==1.7.2
* livelossplot==0.5.5
* matplotlib==3.1.3
* plotly==5.8.0
* seaborn==0.10.0
* sacred==0.8.2
```

All of these can be installed using any package manager such as `pip` or Conda. We recommend using a virtual environment before installing these packages.

### Sample Code

#### Installing the Distance Learner

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
