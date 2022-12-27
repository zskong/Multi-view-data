clear,clc;
addpath('ClusteringMeasure','Datasets','Tools');

fprintf('Demo of FEMSRL on my_UCI\n')

load('Datasets\my_UCI.mat');
v = 3;
X{1} = X1;
X{2} = X2;
X{3} = X3;
for i = 1:v
    X{i} = X{i}./(repmat(sqrt(sum(X{i}.^2,1)),size(X{i},1),1)+1e-8);
end
k = length(unique(gt));

A = cell(1,v); 

lambda = 0.1; 
numanchor= 3*k;

% numanchor=30 	 lambda=0.100000 
% NMI:0.852996, ACC:0.923500, Purity:0.923500, Fsocre:0.856265, Precision:0.854218, times:39.323383

rng(4396,'twister');

for i=1:v
    [~, A{i}] = litekmeans((X{i})',numanchor,'MaxIter', 100,'Replicates',10);
    A{i}=(A{i})';
end

fprintf('params:\t numanchor=%d \t lambda=%f \n',numanchor, lambda);

tic;
[U_results,V_results,Z_results,pre_result]  = FEMSRL(X,A,gt,lambda);
t=toc;
NMI = nmi(pre_result,gt);
Purity = purity(gt, pre_result);
ACC = Accuracy(pre_result,double(gt));
[Fscore,Precision,~] = compute_f(gt,pre_result);
[AR,RI,~,~]=RandIndex(gt,pre_result);
results_log = [NMI,ACC,Purity,Fscore,Precision,t];
fprintf('result:\tNMI:%f, ACC:%f, Purity:%f, Fsocre:%f, Precision:%f, times:%f\n',results_log);
dlmwrite('./Results/UCI_results_logs.txt',[numanchor, lambda],'-append','delimiter','\t','newline','pc');
dlmwrite('./Results/UCI_results_logs.txt',[results_log],'-append','delimiter','\t','newline','pc','precision','%6.6f');
