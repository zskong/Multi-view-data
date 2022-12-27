function [NMI_Max , NMI_Mean , Accu_Max , Accu_Mean] = MKL_Clustering(bestKer,numclass , Label , maxIter)

Accu = zeros(maxIter , 1);
NMI = zeros(maxIter , 1);

for ik =1:maxIter
    [label2, ~] = KernelKmeans(bestKer, numclass, 'MaxIter', 100, 'Replicates', 5);
%     [label2 , energy2] = knkmeans(bestKer,numclass);
    
    [newIndx] = bestMap(Label , label2);
    newIndx = newIndx(:);
    Accu(ik) = mean(Label==newIndx);
    
    nmi_indx = label2(:);
    [newIndx] = bestMap(Label,nmi_indx);
    NMI(ik) = MutualInfo(Label,newIndx);
end

NMI_Max = max(NMI);
NMI_Mean = mean(NMI);
Accu_Max = max(Accu);
Accu_Mean = mean(Accu);
end

