function CQ = CalculateCQ(varargin)
method = varargin{2};
switch method
    % 极大值减极小值
    case 1
        D = varargin{1};
        cluster_count = varargin{3};
        D_normalized = D ./ repmat( sum(D , 2) , 1, cluster_count);
        samplenum = size(D_normalized , 1);
        CQ = ones(samplenum , 1) - max(D_normalized , [] , 2) *cluster_count/2 + min(D_normalized , [] , 2)*cluster_count/2;
        % 极大值减次大值
    case 2
        D = varargin{1};
        [MinV , IndexV] = min(D , [] , 2);
        samplenum = size(D , 1);
        for ii = 1 : samplenum
            D(ii , IndexV(ii)) = 10^5;
        end
        MinV2 = min(D , [] , 2);
        CQ = ones(samplenum , 1) - MinV2 + MinV;
        % 用信息熵来衡量
    case 3
        D = varargin{1};
        [MinV , IndexV] = min(D , [] , 2);
        samplenum = size(D , 1);
        for ii = 1 : samplenum
            D(ii , IndexV(ii)) = 10^5;
        end
        MinV2 = min(D , [] , 2);
        D = [MinV , MinV2];
        D_normalized = D ./ repmat( sum(D , 2) , 1, 2);
        CQ = -sum( D_normalized .* log2(D_normalized) , 2 ) / log2(2);
    case 4
        D = varargin{1};
        alpha = varargin{4};
        cluster_count = varargin{3};
%         D_normalized = D ./ repmat( sum(D , 2) , 1, cluster_count);
        MM = (ones(size(D)) + D.^2 / alpha).^(- (alpha + 1) / 2);
        MM = MM ./ repmat( sum(MM , 2) , 1, cluster_count);
        CQ = -sum( MM .* log(MM) , 2 ) / log(cluster_count);
end
end
