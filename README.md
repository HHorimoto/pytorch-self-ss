# Self-Supervised Learning (SSL) with Pytorch

## Usage

```bash
$ git clone git@github.com:HHorimoto/pytorch-self-ss.git
$ cd pytorch-self-ss
$ ~/python3.10/bin/python3 -m venv .venv
$ . .venv/bin/activate
$ pip install -r requirements.txt
$ source run.sh
```

## Features

### Self-Supervised Learning (SSL)
I trained a model using self-supervised learning with contrastive learning. 
A CNN trained with self-supervised learning on CIFAR-100 was fine-tuned (ft) and transferred (tl) to CIFAR-10. 
For evaluation, the experiment used 1,000 labeled samples and 49,000 unlabeled samples. 
The table and figure below present the experimental results.

**Comparison**

The table shows that **SSL(ft)** and **SSL(tl)** achieves higher accuracy than Supervised Learning **(Sup)**.

|         | Accuracy | Time (s) |
| ------- | :------: | -------- |
| Sup     |  0.4448  | 56.133   |
| SSL(ft) |  0.4829  | 46.424   |
| SSL(tl) |  0.5072  | 43.463   |

#### Reference
[1] [https://github.com/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/11_cnn_pytorch/14_self_supervised_learning.ipynb](https://github.com/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/11_cnn_pytorch/14_self_supervised_learning.ipynb)