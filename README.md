# HMGCF
## Hyperspectral Multilevel GCN and CNN Feature Fusion for Change Detection

Abstract
Hyperspectral Image Change Detection (HSI-CD) focuses on identifying differences in multi-temporal HSIs. Graph Convolutional Networks (GCN) have demonstrated greater promise than convolutional neural networks (CNNs) in remote sensing, particularly for processing HSIs. This is due to GCN's ability to handle non-Euclidean graph-structured information, as opposed to the fixed kernel operations of CNN based on Euclidean structures. Specifically, GCN operates predominantly on superpixel-based nodes. This paper proposes a method, named Hyperspectral Multilevel GCN and CNN Feature Fusion for Change Detection (HMGCF-CD), that integrates superpixel-level GCN with pixel-level CNN for feature extraction and efficient change detection in HSI. The proposed method utilizes the strengths of both CNN and GCN; the CNN branch focuses on feature learning in small-scale, regular regions, while the GCN branch handles large-scale, irregular regions. This approach generates complementary spectral-spatial features at both pixel and superpixel levels. To bridge the structural incompatibility between the Euclidean-data-oriented CNN and the non-Euclidean-data-oriented GCN, HMGCF introduces a graph encoder and decoder. These elements help in propagating features between image pixels and graph nodes, allowing CNN and GCN to function within an integrated end-to-end framework. HMGCF integrates graph encoding into the network, edge weights, and node representations from training data. This innovation improves node feature learning and adapts the graph more effectively to HSI content. Ablation studies on four datasets reveal that the combination of CNN and GCN branches in the HMGCF framework consistently outperforms existing methods by margins ranging from $0.5\%$ to $2.5\%$. In addition, HMGCF shows significant improvements in both kappa and $F1$ scores in all datasets. These results show that the HMGCF is a major advancement in HSI-CD techniques.

![pipeline_](https://github.com/user-attachments/assets/7587b934-425f-4698-b122-f897a1296595)


## 📄 Citation

If you use this code or refer to our method in your research, please cite the following paper:

**Chhaya Katiyar and Vidya Manian**,  
*Hyperspectral Multilevel GCN and CNN Feature Fusion for Change Detection*,  
IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2024.  
🔗 [IEEE Xplore Link](https://ieeexplore.ieee.org/document/10715660)  
📘 DOI: [10.1109/JSTARS.2024.3479920](https://doi.org/10.1109/JSTARS.2024.3479920)

Bibtex format:

### 📚 BibTeX:
```bibtex
@ARTICLE{10715660,
  author={Katiyar, Chhaya and Manian, Vidya},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  title={Hyperspectral Multilevel GCN and CNN Feature Fusion for Change Detection},
  year={2024},
  volume={17},
  number={},
  pages={19146-19159},
  doi={10.1109/JSTARS.2024.3479920},
  keywords={Feature extraction; Convolutional neural networks; Hyperspectral imaging; Change detection (CD); Graph convolutional network (GCN); Supervised learning; Unsupervised learning; Remote sensing imagery}
}

