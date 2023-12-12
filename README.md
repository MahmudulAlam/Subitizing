<h2 align="center">Subitizing 🍂</h2>
<!-- 🍂 -->
<p align="justify">
<b>Question:</b> How many stars are there? 🌟🌟🌟 How about now? 🌟🌟🌟🌟🌟🌟🌟 Certainly, we can answer the first question without explicitly counting each star, and intuitively by looking at the stars all at once we can confidently say there are 3 stars. However, answering the second question was not as much instantaneous and obvious as the first one. What we experienced is called subitizing, the ability to recognize small counts nearly instantaneously (Kaufman et al. 1949). In other words, as the number of items increases their instantaneous countability decreases. However easy and intuitive this is to a human, recent work has demonstrated that a simple convolutional neural network (CNN) failed to perform subitizing. In this work, following the footsteps of prior cognitive science (CogSci) research, we developed a loss function that improves network subitization ability not only of CNNs but also of vision transformers (ViTs).
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
Along with that, we need the <b>hrr</b> library that implements the vector symbolic architecture called Holographic Reduced Representations (HRR) which is used to develop the hrr loss function.  
</p>

```properties 
pip install hrr --upgrade
```

## Getting Started
In this repository, CNN and ViT networks are trained with both hrr and ce loss functions for subitizing tasks. The code for each of the networks is located in ```cnn/``` and ```vit/``` folders, respectively. Each of the folders contains network, train, predict, and saliency files where names are suffixed by corresponding loss function names. The proposed hrr loss function requires key-value pairs per class which are generated in ```generator.py``` file. The ```dataset.py``` file contains the dataloader and ```utils.py``` has the utility files. 

## Methodology 
<p align="justify">
We re-interpret the logits of CNN and ViT as an HRR vector, which we convert to class predictions by associating with each class its own unique HRR vector. We use the concept of <em>binding</em> and <em>unbinding</em> operations of HRR. For each class, a unique <b>K</b>ey-<b>V</b>alue pair is generated. The network predicts the linked key-value pair, i.e., the bound term. During inference, unbinding is applied to the prediction of the network with each key that returns a predicted value. The predicted value is compared with each generated ground truth value of each class using cosine similarity. The arg max of the similarity score will be the class/count output associated with the input. 
</p>

<p align="justify">
The networks are trained using the images of white circles on a black background. After training, during test time experiments are performed by changing the size, shape, color, and boundary representation of the object to examine the subitizing generalizability of each method. The results of the four experiments for both CNN and ViT are shown in the following tables. In each of the four experiments accuracy of the models trained with <b>HRR</b> and <b>CE</b> loss is presented. The HRR-based loss appears to improve the results, especially toward higher subitizing generalization. ViT performed comparatively worse than CNN, however, in general, ViT with HRR loss shows better generalization. In one case of CNN, HRR’s performance has degraded, but still non-trivial performance, and in one case both the HRR loss and CE loss have degenerated worse-than-random guessing. In the case of ViT, HRR’s effectiveness in generalization remains consistent particularly in ‘white rings’ where it outperformed CE over a big margin ranging from 4% to 58%.
</p>

<p align="center">
<img src="https://github.com/MahmudulAlam/Subitizing/assets/37298971/9fb9cb17-a76e-4b26-a6c1-a7568caa36cc" width="800">
</p>

## Saliency Maps 
<p align="justify">
In all four experiments, the saliency maps for both CE and HRR loss are observed and analyzed. In all of the cases, HRR puts more attention toward the boundary regions whereas the network trained with CE loss puts attention on both the inside and output of objects along with the boundary regions. Sample images of the boundary representation test and the corresponding saliency maps for both of the loss functions are given in the following figure. In all cases with the CE loss, we can see spurious attention placed on empty regions of the input - generally increasing in magnitude with more items. By contrast, the HRR loss keeps activations focused on the actual object edges and appears to suffer only for large n when objects are placed too close together.
</p>

<p align="center">
<img src="https://github.com/MahmudulAlam/Subitizing/assets/37298971/b4aa54b3-1ecd-4e62-af8b-e71341c8a615" width="450">
</p>

Moreover, based on the observation of saliency maps of correct and incorrect predictions following conclusions are made:

<ul> 
<li>Even when the CE-based model is correct, its saliency map indicates it uses the inside region of an object and the area around the object/background toward its prediction in almost all cases.</li>
<li>When the HRR-based model is correct, the edges of the objects in the saliency map are usually nearly complete, and large noisy activations can be observed surrounding the boundary regions.</li>
<li>When the CE-based model is incorrect, it often has two objects that are near each other.</li>
<li>When this happens, the CE saliency map tends to produce especially large activations between the objects, creating an artificial "bridge" between the two objects.</li>
<li>When the HRR-based loss is incorrect, it tends to have a saliency map that is either 1) activating on the inside content of the object, or 2) has large broken/incomplete edges detected for the object.</li>
</ul>

<p align="justify">
<b>Bonus:</b> In the beginning, the title of the readme has a leaf emoji. Without looking at that now, how many leaves were there? 
</p>
