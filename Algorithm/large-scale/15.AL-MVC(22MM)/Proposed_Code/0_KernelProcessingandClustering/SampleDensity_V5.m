function [DD , MinD] = SampleDensity_V5(KH , CluNum)
[SNum , ~ , KerNum] = size(KH);
DD = zeros(KerNum , SNum);

for ii = 1 : KerNum
    [U , ~]= mykernelkmeans(KH(: ,: ,ii) , CluNum);
    [~, ~, ~, ~, D] = litekmeans(U , CluNum, 'MaxIter', 100, 'Replicates',30);
    DD(ii , :) = CalculateCQ(D , 3);
end
MinD = min(DD);
end