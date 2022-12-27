% %% This function centerlize the kernel in the fashion as [1]
function [CenteredKernel]= KernelMatrixCentering(KM)
CenteredKernel = zeros(size(KM));
[TestSampleNum , TrainSampleNum,  BaseKernelNum] = size(KM);

VecTest = ones(TestSampleNum,1);
VecTest = sparse(VecTest);
VecTrain = ones(TrainSampleNum,1);
VecTrain = sparse(VecTrain);

I4Trai = speye(TrainSampleNum);
I4Test = speye(TestSampleNum);

TestTransMat = I4Test - (VecTest*VecTest')/TestSampleNum;
TraiTransMat = I4Trai - (VecTrain*VecTrain')/TrainSampleNum;


for BKNum = 1 : BaseKernelNum
    CenteredKernel(: ,: ,BKNum) = TestTransMat * KM(: , :, BKNum) * TraiTransMat;
end
end

% Sihang Zhou

% [1] Algorithms for Learning Kernels Based on Centered Alignment

%% 底下的这一部分是考虑了训练样本的中心化和测试样本中心化两种情况的。测试时所得核矩阵不是方形的需要特殊考虑。
% %% This function centerlize the kernel in the fashion as [1]
% function [CenteredKernel , TotalMeanOfRow , TotalMeanOfColumn , TotalMeanOfAll]= KernelMatrixCentering(KM  , TotalMeanOfRow, TotalMeanOfColumn , TotalMeanOfAll, pos)
%
% SampleNum = size(KM , 1);
% % TransMatrix = eye(SampleNum) - 1/SampleNum*ones(SampleNum,1)*ones(1,SampleNum);
% % CenteredKernel = TransMatrix * KM * TransMatrix;
% BaseKernelNum = size(KM , 3);
% CenteredKernel = zeros(size(KM));
% [row , column] = size(KM{1});
%
% if nargin == 1
%     %% Training condition
%     TotalMeanOfRow = zeros(SampleNum , BaseKernelNum);
%     TotalMeanOfColumn = zeros(BaseKernelNum , SampleNum);
%     TotalMeanOfAll = zeros(BaseKernelNum , 1);
%     for IBKNum  = 1 : BaseKernelNum
%         TempK = KM(: ,: ,IBKNum);
%         TotalMeanOfRow(: , IBKNum) = mean(TempK , 2);
%         TotalMeanOfColumn(IBKNum , :) = mean(TempK , 1);
%         TotalMeanOfAll(IBKNum) = mean(mean(TempK));
%         CenteredKernel(: ,: ,IBKNum) = TempK - repmat(TotalMeanOfRow(: , IBKNum) , [1 column]) - repmat(TotalMeanOfColumn(IBKNum , :) , [row , 1])...
%             + TotalMeanOfAll(IBKNum)*ones(size(KM));
%     end
% else
%     %% Testing condition
%     TotalMeanOfRow = TotalMeanOfRow(pos , :);
%     TotalMeanOfColumn = TotalMeanOfColumn(: , pos);
%     for IBKNum  = 1 : BaseKernelNum
%         TempK = KM(: ,: ,IBKNum);
%         CenteredKernel(: ,: ,IBKNum) = TempK - repmat(TotalMeanOfRow(: , IBKNum) , [1 column]) - repmat(TotalMeanOfColumn(IBKNum , :) , [row , 1])...
%             + TotalMeanOfAll(IBKNum)*ones(size(KM));
%     end
% end
%
%
% end