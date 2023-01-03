% This code runs the SMC algorithm on DBLP dataset.

% The method is based on LMVSC, the core of LMVSC is encapsulated in an independent matlab function.
% Visit lmv.m directly, if you want to learn the details of its implementation.

clear;
clc;

addpath('./dataset');

% load data and graph
load('DBLP4057_GAT_with_idx.mat');

X = double(features);

% label = label.';
[ ~ , argmax ] = max(label,[],2);
y = argmax;
% disp(y);

A{1} = net_APCPA; % 5,000,495
A{2} = net_APTPA; %  6,776,335
A{3} = net_APA; % 11,113

nv = length(A);

ns = length(unique(y));

% Parameter 1: adaptive filter parameter (tunable)
hyperparamter = [0.8];

% Parameter 2: number of anchors (tunable)
numanchor = [95]; % 87 94 95

% Parameter 3: importantnode (tunable)
importantnode = [2];

% Parameter 4: alpha (tunable)
alpha = [20];

for e=1:length(hyperparamter)
    for j=1:length(numanchor)
        for k=1:length(importantnode)
            for i = 1:nv
                A{i} = (A{i} + A{i}')/2;
                D{i} = diag(sum(A{i}));
                % graph filtering 
                X_hat{i} = (eye(size(X,1)) - hyperparamter(e) * (eye(size(X,1)) - D{i}^(-1/2)*A{i}*D{i}^(-1/2))) * X;
            end
            
            % importance_sampling    
            ind = node_sampling_dblp(A, numanchor(j), importantnode(k));
            for i=1:nv
                H{i} = X_hat{i}(ind,:);
                % disp(size(H{i}));
            end
        
            for i=1:length(alpha)
                fprintf('params:\thyperparamter=%f\t\tnumanchor=%d\t\timportantnode=%d\t\talpha=%f\n',hyperparamter(e), numanchor(j), importantnode(k), alpha(i));
                tic;

                % Conduct subspace clustering
                [F,ids] = lmv(X_hat',y,H,alpha(i));
                
                % Performance evaluation of clustering result
                result=ClusteringMeasure(ids,y);
                t=toc;
                fprintf('result:\tACC=%.6f Fscore=%.6f NMI=%.6f ARI=%.6f Purity=%.6f TIMES=%.6f\n',[result t]);

            end
        end
    end
end