This package is implementing the method on Neural Computing and Applications paper: Scalable Multi-view Clustering with Graph Filtering. Please contact [Zkang@uestc.edu.cn](mailto:Zkang@uestc.edu.cn) if you have any questions.

# Data

The DBLP dataset used in the paper can be download here [Google Driver](https://drive.google.com/drive/folders/1zFHrqP7sJiALdIzihmNTvdt0eb5UUGSq?usp=sharing). The required data files should be put into **dataset/** folder.

# Run 

run_smc_dblp.m : We provide a demo for DBLP dataset. If you want to change the default configuration, you can edit this file including dataset, adaptive filter parameter, number of anchors and so on. 


# Note

node_sampling_dblp.m and lower_bound: These files are used for important node sampling.

lmv.m : This file includes multi view subspace clustering.

ClusteringMeasure : This function measures the performance of SMC with 4 indicators -- Accuracy, F1, NMI and ARI.

litekmeans : Perform K-Means clustering.

mySVD : Perform singular value decomposition.

dataset(dir) : Directory for datasets.






