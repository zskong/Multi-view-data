%Clustering on MNIST (70000 samples with 784 dimensions)
clear;
load('Orig.mat'); 

%Clustering using kmeans
rand('twister',5489) 
tic;res=litekmeans(fea,10,'MaxIter',100,'Replicates',10);toc
%Elapsed time is 137.741150 seconds.
res = bestMap(gnd,res);
AC = length(find(gnd == res))/length(gnd)
MIhat = MutualInfo(gnd,res)
%AC: 0.5389
%MIhat:  0.4852

%Clustering using landmark-based spectral clustering
rand('twister',5489) 
tic;res = LSC(fea, 10);toc
%Elapsed time is 20.865842 seconds.
res = bestMap(gnd,res);
AC = length(find(gnd == res))/length(gnd)
MIhat = MutualInfo(gnd,res)
%AC: 0.7270
%MIhat:  0.7222

opts.r = 2;
opts.kmMaxIter = 3;
rand('twister',5489) 
tic;res = LSC(fea, 10, opts);toc
%Elapsed time is 15.471343 seconds.
res = bestMap(gnd,res);
AC = length(find(gnd == res))/length(gnd)
MIhat = MutualInfo(gnd,res)
%AC: 0.7585
%MIhat:  0.7437