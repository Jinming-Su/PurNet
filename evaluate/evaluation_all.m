function [MAE]=eval_MAE
clc
clear

fid = fopen(['evaluation.txt'], 'a+');

dataset = 'ECSSD'
path_output = fullfile(['../test/test_code/result/']);
path_target = fullfile(['../../dataset/' dataset '/groundtruth']);

imnames=dir(path_output);  
imnames2=dir(path_target); 

row_num=length(imnames);
img_num = row_num - 2;
score_MAE = 0;
score_Fbeta = 0;
parfor j=1:img_num
gt=imread(fullfile(path_target, imnames2(j+2).name));
map=imread(fullfile(path_output, imnames2(j+2).name));
map = mat2gray(map(:,:,1));
gt = mat2gray(gt);
gt = (gt == 1);

size_map = size(map);
size_gt = size(gt);

if all(size_gt == size_map) == 0
    map = imresize(map, [size_gt(1) size_gt(2)], 'nearest');
end

score_MAE = score_MAE + computeMAE(map, gt);
score_Fbeta = score_Fbeta + computeFbeta(map, gt);
if mod(j+2, 100) == 0
    fprintf('%s: %d\n', dataset, j+2);
end
end
score_MAE = score_MAE / img_num;
score_Fbeta = score_Fbeta / img_num;

% %%% Fb
belt2=0.3;
reca = zeros(img_num,1);
prec = zeros(img_num,1);
Fm_ans = 0.0;
Fm_gray_i = 1;
parfor i=1:img_num
    target=imread(fullfile(path_target, imnames2(i+2).name));
    output=imread(fullfile(path_output, imnames2(i+2).name));
    target = double(target);
    output = double(output);
    
    output = output(:,:,1);
    mark = mat2gray(target);
    
    output = (output - min(min(output))) * 1.0/(max(max(output)) - min(min(output))) * 255;
    
    size_map = size(output);
    size_gt = size(mark);     
    if all(size_gt == size_map) == 0
        output = imresize(output, [size_gt(1) size_gt(2)], 'nearest');
    end
    
    shape = size(mark);
    label=reshape(mark,1,shape(1) * shape(2));
    score=reshape(output(:,:,1),1,shape(1) * shape(2)); 
    thresh=2*mean2(score);
    if thresh > 255
        thresh = 255;
    end
    sco_th0=(score)>=floor(thresh);
    sco_th=uint8(sco_th0);
    TP = length(find((label == 1) & (sco_th == 1)));
    FP = length(find((label == 0) & (sco_th == 1)));
    FN = length(find((label == 1) & (sco_th == 0)));
    reca(i,1) = TP/(TP+FN+eps);
    prec(i,1) = TP/(TP+FP+eps); 
    if TP+FN == 0 || TP+FP==0
        disp([path_target imnames2(i+2).name]);
    end
    if mod(i+2, 100) == 0
        fprintf('%s wF: %d\n', dataset, i+2);
    end
end

P=mean(prec);
R=mean(reca);
Fmeasure=((1+belt2)*P*R)/(belt2*P+R);
fprintf(fid, '%s: %.6f %.6f %.6f\n', dataset, score_MAE, score_Fbeta, Fmeasure);

end
