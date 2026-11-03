clc; clear; close all

%% Change yolo format to MATLAB format and save it into workspace
folderPath = '../../dataset/RGB data set/RGB_dataset_fix';

classes = {"person", "car", "bus", "truck", "motorcycle", "train", "airplane"};  % Define the classes for detection

testData = yoloformat2trainingData(folderPath,classes,false);

save('testData_x_tal.mat','testData');



%%  
load("Data/testData_x_tal.mat");                               % load data

imds = imageDatastore(testData.imagesPath);                 % path to image data store
blds = boxLabelDatastore(testData(:,2:end));                % all rows, and 2-end
dsVal   = combine(imds, blds);                            


len = length(imds.Files);                                   % length of files
classes = string(testData.Properties.VariableNames(2:end)); % All classes           
detector = yoloxObjectDetector("large-coco");


%% Test detector
% prepare a table with 3 columns - bbox, scores, labels
pred = table('Size',[len 3], ...
    'VariableTypes',{'cell','cell','cell'}, ...
    'VariableNames',{'Boxes','Scores','Labels'});

dt = zeros([len 1]);
reverseStr = '';
%classes=["person", "car", "bus", "truck", "motorbike","motorcycle", "train", "aeroplane", "airoplane", "plane"];
for i=1:len
    I = readimage(imds, i); % read image
    tic
    [bboxes, scores, labels] = detect(detector,I,Threshold=0.5); % detect classes in image
    dt(i) = toc;
    idx = ismember(string(labels)', classes);
    % if  (idx(5)||idx(7))
    % imshow(I)
    % end  
    pred.Boxes{i} = bboxes(idx,:);
    pred.Scores{i} = scores(idx,:);
    pred.Labels{i} = categorical(labels(idx,:), classes);

    %
    msg = sprintf('Progress: %.2f%%', (i/len)*100);
    fprintf(reverseStr);                                % delete line
    fprintf('%s',msg);                                  % write msg
    reverseStr = repmat('\b', 1, length(msg));          % prepare delte for next intration
end

clearvars -except pred blds imds detector dt
%% Get metrics
metrics = evaluateObjectDetection(pred, blds);

% Performance check
% [cm, confClassNames] = confusionMatrix(metrics);
% chart = confusionchart(cm{:},confClassNames);
% title('Object Detection Confusion Matrix');
%% F1 score caculate

% Extract precision and recall
[recall, precision, ~] = precisionRecall(metrics);

% Create cell function that caculate F1-score
f1Scores = cellfun(@(p, r) (2 .* p .* r) ./ (p + r + eps), ...
                   precision, recall, 'UniformOutput', false);

% Extract the maximum F1-score
maxF1PerClass = cellfun(@max, f1Scores);

% Get class names
classNames = metrics.ClassNames;

% Create a struct that shows f1 scores for each class
f1 = struct('className',classNames,'maxF1',maxF1PerClass);

%% draft
len = length(blds.LabelData);
% [r,~] = find(cell2mat(metrics.ImageMetrics.AP) < 0.7); % num->r(num)
% len = length(r);

figure;
for num=1:len
    I = readimage(imds,num);
    
    % Display the first image with detected bounding boxes
    LabelScoreStr = compose("%s-%.2f",pred.Labels{num},pred.Scores{num});
    detectedImage = insertObjectAnnotation(I, 'rectangle', pred.Boxes{num}, LabelScoreStr,"LineWidth",3,'FontSize',25);
    groundTruthImage = insertObjectAnnotation(I, 'rectangle', blds.LabelData{num,1},blds.LabelData{num,2},"LineWidth",3,'FontSize',25);
    
    subplot(121)
    imshow(detectedImage);
    title(sprintf('detected Image image %d',num));
    subplot(122)
    imshow(groundTruthImage);
    title(sprintf('ground true bbox image %d',num));
    pause;
end


%% Save the YOLOv4 detector, metrics, and predictions into a structure

yolox = struct('Detector', detector, ...
                'Metrics', metrics, ...
                'Predictions', pred, ...
                'f1Scores',f1, ...
                'AvgProcessTime',mean(dt));

save('yolox_results.mat', 'yolox');


