# NeuronFair

#### Description

Deep neural networks (DNNs) have demonstrated their outperformance in various domains. However, it raises a social concern whether DNNs can produce reliable and fair decisions especially when they are applied to sensitive domains involving valuable resource allocation, such as education, loan, and employment. It is crucial to conduct fairness testing before DNNs are reliably deployed to such sensitive domains, i.e., generating as many instances as possible to uncover fairness violations. However, the existing testing methods are still limited from three aspects: interpretability, performance, and generalizability. To overcome the challenges, we propose NeuronFair, a new DNN fairness testing framework that differs from previous work in several key aspects: (1) interpretable - it quantitatively interprets DNNs’ fairness violations for the biased decision; (2) effective - it uses the interpretation results to guide the generation of more diverse instances in less time; (3) generic - it can handle both structured and unstructured data. Extensive evaluations across 7 datasets and the corresponding DNNs demonstrate NeuronFair’s superior performance. For instance, on structured datasets, it generates much more instances (∼×5.84) and saves more time (with an average speedup of 534.56%) compared with the state-of-the-art methods. Besides, the instances of NeuronFair can also be leveraged to improve the fairness of the biased DNNs, which helps build more fair and trustworthy deep learning systems. 

#### Installation

This project is based on python 3.6. 

Please install the following:

1.  numpy ==1.19.5
2.  tensorflow==1.14.0
3.  tflearn==0.3.2
4.  aif360 ==0.4.0

#### Instructions
>NeuronFair
>>LICENSE

>>README.en.md

>>clusters

>>datasets

>>models

>>nf_data

>>nf_model

>>>dnn_models.py

>>>layer.py

>>> model.py

>>>model_train.py

>>>network.py

>>output

>>src

>>>dnn_nf.py

>>>nf_utils.py

>>utils

>>>config.py

>>>utils.py

>>>utils_tf.py

The folders **datasets** and **models** store the original data sets and the models generated by training, respectively. The folder **clusters** mainly stores the clustered datasets. Data preprocessing code is in folder **nf_data**. The model structure code is in the folder **nf_model**. Folder **output** is used to store fairness testing results. The fairness testing code is located in **src/dnn_nf.py**.

#### Getting Started

1.  train

```python
# param dataset: the name of testing dataset
# param model_path: the path to save trained model
# param nb_epochs: Number of epochs to train model
# param batch_size: Size of training batches
# lparam earning_rate: Learning rate for training
python nf_model/model_train.py 'census' "../models/" 1000 64 0.001
```

2.  fairness testing
```python
dataset, sensitive_param, model_path, cluster_num, max_global, max_local, max_iter, ReLU_name
# param dataset: the name of testing dataset
# param sens_name: the name of sensitive feature 
# param sens_param: the index of sensitive feature, index start from 1, 9 for gender, 8 for race
# param model_path: the path to save trained model
# param cluster_num: the number of clusters to form as well as the number of centroids to generate
# param max_global: maximum number of samples for global search 
# param max_local: maximum number of samples for local search
# param max_iter: maximum iteration of global perturbation
# param ReLU_name: the name of bias layer of dnn model
python src/dnn_nf.py "census" "gender" 9 '../models/census/dnn/best.model' 4 1000 1000 40 "ReLU5"
```

The generated testing instances will be stored in the output.

#### Citation

A technical description of CertPri is available in this
[paper](https://dl.acm.org/doi/abs/10.1145/3510003.3510123). Below is the bibtex entry for this paper.
If you find this code useful for your research, please consider citing:

```
@inproceedings{Zheng2022NeuronFair,
   author = {Zheng, Haibin and Chen, Zhiqing and Du, Tianyu and Zhang, Xuhong and Cheng, Yao and Ji, Shouling and Wang, Jingyi and Yu, Yue and Chen, Jinyin},
   title = {NeuronFair: Interpretable White-Box Fairness Testing through Biased Neuron Identification},
   booktitle = {44th International Conference on Software Engineering},
   address = {New York, NY, USA},
   pages = {1-13},
   date = {May 21–29},
   publisher = {{ACM}},
   year = {2022}
}
```

