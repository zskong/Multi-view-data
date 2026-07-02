<div align="center">

# 📚 Multi-View Datasets Repository
**A curated collection of source code and datasets for Multi-view Learning & Clustering.**

[![Datasets Count](https://img.shields.io/badge/Counts-96-blue.svg)](#)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](#)

</div>

---

## 📖 Introduction
This repository is maintained as a comprehensive collection of source code and benchmark datasets for multi-view clustering research.

> ⚖️ **Rights**: Explanation and maintenance rights belong to the author. 

---

## 🗺️ Dataset Navigation
Jump to datasets based on sample size:
* [🟢 Small-scale (< 1,000)](#-small-scale-datasets)
* [🟡 Medium-scale (1,000 - 10,000)](#-medium-scale-datasets)
* [🔴 Large-scale (> 10,000)](#-large-scale-datasets)

---

## 🟢 Small-scale Datasets
*Sample size < 1,000*

| Dataset | Samples | Views | Clusters | Dimensions | Source | Note |
| :--- | :---: | :---: | :---: | :--- | :---: |:---: |
| **<br>CESC</br>(Cervical squamous cell carcinoma)** | 124 | 4 | 3 | 2000/2000/311/219 | [Link](https://www.nature.com/articles/s41467-022-35031-9) | <br>Multi-omics</br>(mDNA,RNA,miRNA,RPPA) |
| **Yale** | 165 | 3 | 15 | 4096/3304/6750 | [Link](http://cvc.cs.yale.edu/cvc/projects/yalefaces/yalefaces.html) |
| **3-Sources ⭐** | 169 | 3 | 6 | 3560/3631/3068 | [Link](http://mlg.ucd.ie/datasets/3sources.html) |
| **TwoMoon** | 200 | 2 | 2 | 2/2 | Synthetic dataset |
| **webkb** | 203 | 3 | 4 | 1703/230/230 | [Link](https://linqs.soe.ucsc.edu/data) |
| **Sonar** | 208 | 3 | 2 | 20/20/20 | |
| **MSRC** | 210 | 5 | 7 | 24/576/512/256/254 | [Link](https://www.cnblogs.com/picassooo/p/12890078.html) |
| **MSRCV1** | 210 | 6 | 7 | 1302/48/512/100/256/210 | [Link](https://www.cnblogs.com/picassooo/p/12890078.html) |
| **GBM (Glioblastoma multiforme)** | 248 | 3 | 4 | 534/5000/12042 | [Link](https://link.springer.com/article/10.1007/s13755-024-00274-x) | Multi-omics(Gene expression,miRNA,DNA) |
| **LGG (Lower gradeg lioma)** | 267 | 4 | 3 | 2000/2000/333/209 | [Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9358980) | Multi-omics |
| **ThreeRing** | 300 | 2 | 3 | 2/2 | Synthetic dataset |
| **Dermatology** | 366 | 2 | 6 | 11/22 | [Link](https://archive.ics.uci.edu/dataset/33/dermatology) |
| **BRCA (Breast adenocar cinoma)**| 398 | 4 | 4 |  2000/2000/278/212 | [Link](https://www.nature.com/articles/s41586-021-04278-5) | Multi-omics |
| **ORL** | 400 | 3 | 40 | 4096/3304/6750 | [Link](https://gitee.com/zhangfk/multi-view-dataset) |
| **ORL** | 400 | 4 | 40 | 512/59/864/254 | [Link](https://gitee.com/zhangfk/multi-view-dataset) |
| **NGs ⭐** | 500 | 3 | 5 | 2000/2000/2000 | [Link](http://ligmembres.imag.fr/grimal/data.html) |
| **20newsgroups** | 500 | 3 | 5 | 2000/2000/2000 | [Link](http://ligmembres.imag.fr/grimal/data.html) |
| **Caltech101_3view** | 512 | 3 | 11 | 254/512/36 | [Link](https://hyper.ai/datasets/5258) |
| **Forest** | 523 | 2 | 4 | 9/18 | [Link](https://archive.ics.uci.edu/dataset/333/forest+type+mapping) |
| **BBCSport ⭐** | 544 | 2 | 5 | 3183/3203 | [Link](http://mlg.ucd.ie/datasets/segment.html) |
| **Notting-Hill ⭐** | 550 | 3 | 5 | 2000/3304/6750 | [Link](https://ieeexplore.ieee.org/document/6619294/) |
| **Prokaryotic** | 551 | 3 | 4 | 438/3/393 | [Link](http://lin-group.cn/database/ppd/index.php) |
| **Reuters** | 600 | 5 | 6 | Need to deal | [Link](http://lig-membres.imag.fr/grimal/data.html) |
| **synthetic3d** | 600 | 3 | 3 | 3/3/3 | Synthetic dataset |
| **CUB** | 600 | 2 | 10 | 1024/300 | [Link](https://papers.nips.cc/paper/2019/file/11b9842e0a271ff252c1903e7132cd68-Paper.pdf) |
| **Movie** | 617 | 2 | 17 | 1878/1398 | [Link](https://lig-membres.imag.fr/grimal/data.html) |
| **YaleB** | 650 | 3 | 10 | 2500/3304/6750 | [Link](http://cvc.cs.yale.edu/cvc/projects/yalefacesB/yalefacesB.html) |
| **BBC4view_685** | 685 | 4 | 5 | 4659/4633/4665/4684 | [Link](http://mlg.ucd.ie/datasets/segment.html) |
| **WikipediaArticles** | 693 | 2 | 10 | 128/10 | [Link](https://github.com/wangsiwei2010/large_scale_multi-view_clustering_datasets) |
| **ProteinFold** | 694 | 12 | 27 | 27/.../27 | [Link](https://github.com/wangsiwei2010/large_scale_multi-view_clustering_datasets) |
| **Oxford** | 800 | 3 | 4 | 1764/10/128| [Link](https://ieeexplore.ieee.org/document/6248092) |
---

## 🟡 Medium-scale Datasets
*Sample size 1,000 - 10,000*

| Dataset | Samples | Views | Clusters | Dimensions | Source | Note |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: |
| **WebKB2** | 1051 | 2 | 2 | 2949/334 | [Link](https://linqs.soe.ucsc.edu/data) |
| **Reuters** | 1200 | 5 | 6 | 2000/2000/2000/2000/2000 | [Link](https://github.com/dugzzuli/A-Survey-of-Multi-view-Clustering-Approaches) |
| **Flower17 ⭐** | 1360 | 7 | 17 | 1360/.../1360 | [Link](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/) |
| **COIL20_pca** | 1440 | 3 | 20 | 30/19/30 | [Link](http://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php) |
| **COIL20 ⭐** | 1440 | 3 | 20 | 4096/3304/6750 | [Link](http://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php) |
| **FuCOIL20** | 1440 | 3 | 20 | 1024/1024/324 | [Link](http://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php) |
| **RGB-D ⭐** | 1449 | 2 | 13 | 2048/300 | [Link](https://github.com/DanielTrosten/mvc) |
| **Caltech101-7 ⭐** | 1474 | 6 | 7 | 48/40/254/1984/512/928 | [Link](http://www.vision.caltech.edu/ImageDatasets/Caltech101/) |
| **GRAZ02** | 1476 | 6 | 4 | 512/32/256/500/500/680 | [Link](http://www.emt.tugraz.at/~pinz/data/GRAZ_02) |
| **Reuters_21578** | 1500 | 5 | 6 | 21531/24892/34251/15506/11547 | [Link](https://archive.ics.uci.edu/ml/datasets/reuters-21578+text+categorization+collection) |
| **Youtube** | 1592 | 2 | 11 | - | [Link](https://www.kaggle.com/datasnaek/youtube-new) |
| **100Leaves** | 1600 | 3 | 100 | 64/64/64 | [Link](https://archive.ics.uci.edu/ml/datasets/One-hundred+plant+species+leaves+data+set) |
| **UCI-Digits ⭐** | 2000 | 3 | 10 | 64/76/216 | [Link](http://archive.ics.uci.edu/ml/datasets) |
| **HW2sources** | 2000 | 2 | 10 | 786/256 | [Link](http://archive.ics.uci.edu/ml/datasets/Multiple+Features) |
| **Handwritten** | 2000 | 6 | 10 | 64/76/216/6/240/47 | [Link](http://archive.ics.uci.edu/ml/datasets/Multiple+Features) |
| **Mfeat** | 2000 | 6 | 10 | 64/76/216/6/240/47 | [Link](https://archive.ics.uci.edu/ml/datasets/Multiple+Features) |
| **NUS_WIDE** | 2000 | 5 | 31 | 65/226/145/74/129 | [Link](https://github.com/youweiliang/Multi-view_Graph_Learning/tree/master/data) |
| **MNIST** | 2000 | 3 | 10 | 30/9/30 | [Link](https://github.com/032004129xuzhiyong/GCNII/tree/27a8717c1174883deb00eed766a15624b7bc2aa0/data) |
| **LandUse-21 ⭐** | 2100 | 3 | 21 | 20/59/40 | [Link](https://hyper.ai/datasets/5431) |
| **Caltech101-20 ⭐** | 2386 | 6 | 20 | 48/40/254/1984/512/928 | [Link](http://www.vision.caltech.edu/ImageDatasets/Caltech101/) |
| **YaleB_Extend** (visualization) | 2424 | 5 | 38 | 1024/1024/1024/1024/1024 | [Link](https://github.com/032004129xuzhiyong/GCNII/tree/27a8717c1174883deb00eed766a15624b7bc2aa0/data) |
| **NUS** | 2400 | 6 | 12 | 64/144/73/128/225/500 | [Link](https://dl.acm.org/doi/10.1145/1646396.1646452) |
| **2V_BDGP** | 2500 | 2 | 5 | 1750/79 | [Link](https://ranger.uta.edu/heng/Drosophila) |
| **BDGP_fea** | 2500 | 3 | 5 | 1000/500/250 | [Link](https://ranger.uta.edu/heng/Drosophila) |
| **Toydata_5** (visualization) | 2500 | 2 | 5 | 2/2 | Synthetic dataset |
| **Scene** | 2688 | 4 | 8 | 512/432/256/48 | [Link](https://mvrl.cse.wustl.edu/datasets/amos) |
| **Cora** | 2708 | 2 | 7 | 1433/2708 | [Link](https://github.com/032004129xuzhiyong/GCNII/tree/27a8717c1174883deb00eed766a15624b7bc2aa0/data) |
| **Wiki_fea** | 2866 | 2 | 10 | 128/10 | [Link](https://dumps.wikimedia.org/zhwiki/latest/) |
| **Toydata_3** (visualization) | 3000 | 2 | 3 | 2/2 | Synthetic dataset |
| **CiteSeer** | 3312 | 2 | 6 | 3312/3703 | [Link](http://lig-membres.imag.fr/grimal/data.html) |
| **ImageNet** | 4000 | 3 | 4 | 1764/10/128 | [Link](https://www.sciencedirect.com/science/article/abs/pii/S0031320326003900) |
| **Scene15 ⭐** | 4485 | 3 | 15 | 20/59/40 | [Link](https://figshare.com/articles/dataset/15-Scene_Image_Dataset/7007177) |
| **NH_p4660** | 4660 | 3 | 5 | 2000/3304/6750 | [Link](https://ieeexplore.ieee.org/document/6619294/) |
| **2V_MNIST_USPS** (visualization) | 5000 | 2 | 10 | 784/784 | [Link](http://yann.lecun.com/exdb/mnist) |
| **MITIndoor** | 5360 | 4 | 67 | 1770/3600/1240/4096 | [Link](http://web.mit.edu/torralba/www/indoor.html.) |
| **VOC ⭐** (PASCAL VOC 2007) | 5649 | 2 | 20 | 512/399 | [Link](https://github.com/DanielTrosten/mvc) |
| **CCV ⭐** | 6773 | 3 | 20 | 20/20/20 | [Link](https://www.ee.columbia.edu/ln/dvmm/CCV/) |
| **Caltech101-all** | 8677 | 4 | 101 | 3540/4800/1240/2048 | [Link](https://ieeexplore.ieee.org/abstract/document/1384978) |
| **Caltech101-all_fea ⭐** | 9144 | 5 | 102 | 48/40/254/512/928 | [Link](http://www.vision.caltech.edu/ImageDatasets/Caltech101/) |
| **Fashion** (visualization) | 10000 | 3 | 10 | 784/784/784 | [Link](https://github.com/zalandoresearch/fashion-mnist) |
| **MNIST** (small dimension) | 10000 | 3 | 10 | 30/9/30 | [Link](https://github.com/dugzzuli/A-Survey-of-Multi-view-Clustering-Approaches) |
| **Hdigit** | 10000 | 2 | 10 | 784/256 | - |
| **Mfeat** | 10000 | 2 | 10 | 784/256 | [Link](https://archive.ics.uci.edu/ml/datasets/Multiple+Features) |
| **CIFAR10_deep** | 10000 | 4 | 10 | 1000/1000/1000/2048 | - |

---

## 🔴 Large-scale Datasets
*Sample size > 10,000*

| Dataset | Samples | Views | Clusters | Dimensions | Source | Note |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: |
| **SUNRGBD** | 10335 | 2 | 45 | 4096/4096 | - |
| **ALOI100 ⭐** | 10800 | 4 | 100 | 77/13/64/125 | [Link](https://elki-project.github.io/datasets/multi_view) |
| **Animal** | 11673 | 4 | 20 | 2689/2000/2001/2000 | [Link](https://github.com/wangsiwei2010/large_scale_multi-view_clustering_datasets) |
| **STL-10 ⭐** | 13000 | 3 | 10 | 1024/512/2048 | [Link](https://cs.stanford.edu/~acoates/stl10/) |
| **Reuters** | 18758 | 5 | 6 | 21531/24892/34251/15506/11547 | [Link](https://archive.ics.uci.edu/ml/datasets.html) |
| **Cifar10-4** | 20000 | 3 | 4 | 324/10/128 | [Link](https://www.sciencedirect.com/science/article/abs/pii/S0031320326003900) |
| **NUSWIDEOBJ ⭐** | 30000 | 5 | 31 | 65/226/145/74/129 | [Link](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html) |
| **MoisyMNIST ⭐** | 30000 | 2 | 10 | 784/784 | [Link](http://proceedings.mlr.press/v37/wangb15.pdf) |
| **AwA_fea ⭐** | 30475 | 4 | 20 | 2688/2000/252/2000/2000/2000 | [Link](http://cvml.ist.ac.at/AwA2/AwA2-data.zip) |
| **Caltech256_fea** | 30607 | 3 | 257 | 1024/512/2048 | [Link](http://www.vision.caltech.edu/ImageDatasets/Caltech101/) |
| **VGGFace2-50 ⭐** | 34027 | 4 | 50 | 944/576/512/640 | [Link](https://hyper.ai/datasets/5711) |
| **Cifar10-8** | 40000 | 3 | 8 | 324/10/128 | [Link](https://www.sciencedirect.com/science/article/abs/pii/S0031320326003900) |
| **CIFAR10** | 50000 | 3 | 10 | 1024/512/2048 | [Link](http://www.cs.toronto.edu/~kriz/cifar.html) |
| **CIFAR100** | 50000 | 3 | 100 | 1024/512/2048 | [Link](http://www.cs.toronto.edu/~kriz/cifar.html) |
| **Noisy Mnist** | 50000 | 2 | 10 | 784/784 | [Link](http://proceedings.mlr.press/v37/wangb15.pdf) |
| **fmnist ⭐** | 60000 | 3 | 10 | 1280/512/512 | [Link](https://www.worldlink.com.cn/en/osdir/fashion-mnist.html) |
| **MNIST** | 60000 | 3 | 10 | 342/1024/64 \| 784/784 | [Link](https://www.worldlink.com.cn/en/osdir/fashion-mnist.html) |
| **VGGFace4-100** | 72283 | 4 | 200 | 944/576/512/640 | [Link](https://hyper.ai/datasets/5711) |
| **tinyimage** | 100000 | 3 | 200 | 1280/512/512 | [Link](https://paperswithcode.com/dataset/tiny-imagenet) |
| **YoutubeFace ⭐**| 101499 | 5 | 31 | 64/512/64/647/838 | [Link](https://www.cs.tau.ac.il/~wolf/ytfaces/) |
| **YouTubeFace50** | 126054 | 4 | 50 | 944/576/512/640 | [Link](https://www.cs.tau.ac.il/~wolf/ytfaces/) |
| **YouTube** | 152549 | 3 | 65 | 1024/768/1152 | [Link](https://www.cs.tau.ac.il/~wolf/ytfaces/) |

---

## 🔗 Acknowledgements
Special thanks to the following sources for their contributions to the multi-view community:
* [wangsiwei2010/large_scale_multi-view_clustering_datasets](https://github.com/wangsiwei2010/large_scale_multi-view_clustering_datasets)
* [wangsiwei2010/awesome-multi-view-clustering](https://github.com/wangsiwei2010/awesome-multi-view-clustering)
* [zhangfk/multi-view-dataset](https://gitee.com/zhangfk/multi-view-dataset)
