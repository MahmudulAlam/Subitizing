<h2 align="center">Subitizing</h2>

<p align="justify">
While deep learning has enjoyed significant success in computer vision tasks over the past decade, many shortcomings still exist from a cognitive science perspective. In particular, the ability to <em>subitize</em>, quickly and accurately identify the small (≤ 6) count of an item, is not well learned by current convolutional neural networks (CNN) when using a standard cross-entropy loss. Our study demonstrates that adapting tools used in cognitive science research can improve the subitizing generalization of simple CNN. To do so we use the Holographic Reduced Representations (HRR) to implement an alternative loss function. This HRR-based loss improves - but does not solve - a CNN's ability to generalize subitizing.
</p>

### Requirements

- PyTorch ```conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch```

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
