function [H_normalized,obj , Distance]= mykernelkmeans_V2(K , cluster_count , Weight)

Weight = diag(Weight);
K = (K+K')/2;
opt.disp = 0;
[H,~] = eigs(K,cluster_count,'la',opt);
% obj0 = trace(H' * K * H) - trace(K);
% obj1 = -trace(K * (H * H') .* Weight) + trace(K.*Weight);
Distance = diag(K-K*(H*H'));
obj = Distance'*diag(Weight);
Distance = Distance';
H_normalized = H ./ repmat(sqrt(sum(H.^2, 2)), 1,cluster_count);

HH = - sum( H_normalized .* log(H_normalized) );
obj = sum(HH);
%H_normalized = H;

% °×»¯
% C = cov(H_normalized);
% M = mean(H_normalized);
% [V,D] = eig(C);
% P = V * diag(sqrt(1./(diag(D) + 0.1))) * V';
% H_normalized = bsxfun(@minus, H_normalized, M) * P;

% [~, ~, ~, sumD, D] = litekmeans(H_normalized,cluster_count, 'MaxIter', 100, 'Replicates',30);
% 
% obj = sum(sumD) / sum(sum(D));