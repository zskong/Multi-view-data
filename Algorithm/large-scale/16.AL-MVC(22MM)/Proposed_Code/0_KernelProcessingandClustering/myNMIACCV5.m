function [res_mean,res_std,res1,res2,res3]= myNMIACCV5(U,Y,numclass)

stream = RandStream.getGlobalStream;
reset(stream);
anchor_num = size(U,2);
U_normalized = U ./ repmat(sqrt(sum(U.^2, 2)), 1,anchor_num);
maxIter = 100;


indx = litekmeans(U_normalized,numclass, 'MaxIter',100, 'Replicates',maxIter);
indx = indx(:);
[newIndx] = bestMap(Y,indx);
res1 = mean(Y==newIndx);
res2 = MutualInfo(Y,newIndx);
res3= purFuc(Y,newIndx);
    
res_mean(1) = mean(res1);
res_mean(2) = mean(res2);
res_mean(3) = mean(res3);

res_std(1) = std(res1);
res_std(2) = std(res2);
res_std(3) = std(res3);