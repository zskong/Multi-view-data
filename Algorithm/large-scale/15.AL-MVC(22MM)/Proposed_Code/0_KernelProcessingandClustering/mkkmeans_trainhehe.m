% Mehmet Gonen (mehmet.gonen@gmail.com)

function [state] = mkkmeans_trainV3(Km, parameters)
tic;
P = size(Km, 3);
samplenum = size(Km , 1);
theta = ones(P, 1) / P;
K_theta = calculate_kernel_theta(Km, theta.^2);

opt.disp = 0;
objective = zeros(parameters.iteration_count, 1);
for iter = 1:parameters.iteration_count
    fprintf(1, 'running iteration %d...\n', iter);
    [H, ~] = eigs(K_theta, parameters.cluster_count, 'la', opt);
    
    Q = zeros(P, P);
    for m = 1:P
        Q(m, m) = trace(Km(:, :, m)) - trace(H' * Km(:, :, m) * H);
    end
    %         res = mskqpopt(Q, zeros(P, 1), ones(1, P), 1, 1, zeros(P, 1), ones(P, 1), [], 'minimize echo(0)');
    %         theta = res.sol.itr.xx;
    theta = quadprog(Q , zeros(P, 1), [], [], ones(P , 1)' , 1 , zeros(P,1)' , []);
    K_theta = calculate_kernel_theta(Km, theta.^2);
    
    objective(iter) = trace(H' * K_theta * H) - trace(K_theta);
end
H_normalized = H ./ repmat(sqrt(sum(H.^2, 2)), 1, parameters.cluster_count);

stream = RandStream.getGlobalStream;
reset(stream);
% state.clustering = kmeans(H_normalized, parameters.cluster_count, 'MaxIter', 1000, 'Replicates', 10);
[state.clustering, ~, ~, ~, D] = litekmeans(H_normalized,parameters.cluster_count, 'MaxIter', 100, 'Replicates',30);
state.objective = objective;
state.parameters = parameters;
state.theta = theta;
state.time = toc;

% 计算各样本的隶属度
alpha = 3;
CQ = CalculateCQ(D , 2 , parameters.cluster_count);

state.Criteria = CQ;
end

function K_theta = calculate_kernel_theta(K, theta)
K_theta = zeros(size(K(:, :, 1)));
for m = 1:size(K, 3)
    K_theta = K_theta + theta(m) * K(:, :, m);
end
end

function CQ = CalculateCQ(varargin)
method = varargin{2};
switch method
    % 极大值减极小值
    case 1
        D = varargin{1};
        cluster_count = varargin{3};
        D_normalized = D ./ repmat( sum(D , 2) , 1, cluster_count);
        samplenum = size(D_normalized , 1);
        CQ = ones(samplenum , 1) - max(D_normalized , [] , 2) + min(D_normalized , [] , 2);
        % 极大值减次大值
    case 2
        D = varargin{1};
        cluster_count = varargin{3};
        D_normalized = D ./ repmat( sum(D , 2) , 1, cluster_count);
        [MaxV , IndexV] = max(D_normalized , [] , 2);
        samplenum = size(D_normalized , 1);
        for ii = 1 : samplenum
            D_normalized(ii , IndexV(ii)) = 0;
        end
        MaxV2 = max(D_normalized , [] , 2);
        CQ = MaxV - MaxV2;
        % 用信息熵来衡量
    case 3
        D = varargin{1};
        cluster_count = varargin{3};
        D_normalized = D ./ repmat( sum(D , 2) , 1, cluster_count);
        CQ = -sum( D_normalized .* log(D_normalized) , 2 ) / log(cluster_count);
    case 4
        D = varargin{1};
        alpha = varargin{4};
        cluster_count = varargin{3};
        D_normalized = D ./ repmat( sum(D , 2) , 1, cluster_count);
        MM = (ones(size(D_normalized)) + D_normalized.^2 / alpha).^(- (alpha + 1) / 2);
        MM = MM ./ repmat( sum(MM , 2) , 1, cluster_count);
        CQ = -sum( MM .* log(MM) , 2 ) / log(cluster_count);
end
end
