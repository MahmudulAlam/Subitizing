<h2 align="center">Subitizing 🍂</h2>

<p align="justify">
<b>Question:</b> How many stars are there? 🌟🌟🌟 How about now? 🌟🌟🌟🌟🌟🌟🌟 Certainly, we can answer the first question without explicitly counting each star, and intuitively by looking at the stars all at once we can confidently say there are 3 stars. However, answering the second question was not as much instantaneous and obvious as the first one. What we experienced is called subitizing, the ability to recognize small counts nearly instantaneously (Kaufman et al. 1949). In other words, as the number of items increases their instantaneous countability decreases. However much easy and intuitive this is to a human, recent work has demonstrated that simple CNN failed to perform subitizing. In this paper, following the footsteps of prior cognitive science (CogSci) work, we developed a loss function that improves network subitization ability not only of convolutional neural networks (CNN) but also of vision transformers (ViT).
</p>

<p align="justify">
The proposed loss function leverages a vector symbolic architecture called Holographic Reduced Representations (HRR). Although the proposed loss function is not a complete answer to the subitizing question, it does help networks to learn better generalization to subitizing compared to a regular cross-entropy (CE) loss function. Our results are intriguing in that we did not design the HRR loss to be biased toward numerosity or subitizing via symbolic manipulation, but instead defined a simple loss function as a counterpart to the CE loss that retains a classification focus. This may imply some unique benefit to the HRR operator in improving generalization and supports the years of prior work using it for CogSci research.
</p>

## Requirements
![requirements](https://img.shields.io/badge/Python-3.9.16-86f705.svg?longCache=true&style=flat&logo=python)

<p align="justify">
The code is written in <a href=https://pytorch.org>pytorch</a> which is the main dependency to run the code.
</p>

```properties 
conda create -n <env-name> python=3.9.16
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

<p align="justify">
Along with that we need the <b>hrr</b> library that implements the vector symbolic architecture called Holographic Reduced Representations (HRR) which is used to develop the hrr loss function.  
</p>

```properties 
pip install hrr --upgrade
```

## Getting Started
In this repository, CNN and ViT networks are training with both hrr and ce loss functions for subitizing tasks. The code for each of the networks are located in ```cnn/``` and ```vit/``` folders, respectively. Each of the folders contains network, train, predict, and saliency files where names are suffixed corresponding loss function name. The proposed hrr loss function requries key-value pairs per class which are generated in ```generator.py``` file. The ```dataset.py``` file contains the dataloader and ```utils.py``` has the utility files. 

### Methodology 
<p align="justify">
We re-interpret the logits of a CNN as an HRR vector, which we will then convert to a class prediction by associating with each class its own unique HRR vector. We will use the concept of <em>binding</em> and <em>unbinding</em> operations of HRR and the network will predict the linked key-value pair, i.e., the bound term. Unique key, value, and their pair is generated in <em>generator.py</em> which is tested in <em>generator_test.py</em> to check whether it can retrieve the original vector correctly or not.
</p>

<p align="justify">
We trained the network using the images of white circles on a black background which is given in the following figure.
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/37298971/169652699-32621553-a5cc-44b1-8741-a1eeafee2ae3.png" width="600">
</p>

<p align="justify">
Afterward, we performed 4 experiments using the trained network. In the test images, we have made the circle sizes 50% larger, changed circles to triangles and squares, swapped the circle and background color, and represented circles with a boundary region of white rings. The results of these 4 experiments are presented in the following table for both the HRR approach and end-to-end cross-entropy (CE) approach.
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/37298971/169652713-3ef2883b-1332-47db-b562-61bbb77c4679.png" width="600">
</p>

<p align="justify">
Experiments 1 to 4 demonstrate CNN’s lack of generalization in learning. To improve the generalization, instead of learning from single-shaped images, each class is built with different shaped objects. Therefore, learning would be independent of the shape of the object. Moreover, each object is represented by its boundary which bridges the representation of the black object on a white background and the white object on a black background. Samples of these images are given below. 
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/37298971/170328878-39bf5eea-bafa-49bb-b4ea-066505338d86.png" width="600">
</p>

<p align="justify">
The network is re-trained using 80% of the boundary representation figures and the rest 20% of the images are used for testing. The accuracy of a test set of in-distribution is shown in the following table. While the CE appears to obtain better training accuracy, the goal of this study is the generalization of subitizing ability.
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/37298971/169652737-36e2570a-0e9d-4b71-9719-961908fd0533.png" width="600">
</p>

<p align="justify">
Table 3 reveals how the results deteriorate by only changing the scale of the object. However, in the case of scaling up, both of the methods show solid evidence of subitizing, i.e., the accuracy decreases as the number of objects in the image increases. The proposed method has achieved an average accuracy of 49% whereas the CE approach has achieved an average accuracy of 45.6%, but the CE’s performance is inflated in the sense that it has a higher training accuracy and drops precipitously.
</p> 

<p align="center">
<img src="https://user-images.githubusercontent.com/37298971/169652755-32b38438-2dc1-4e66-95cc-a396bb16966c.png" width="600">
</p>
