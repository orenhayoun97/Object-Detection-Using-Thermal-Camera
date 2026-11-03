clc;clear; close all

%% Change yolo format to MATLAB format and save it into workspace
folderPath = '../../dataset/RGB data set/RGB_dataset_fix';
classes = {"person", "car", "bus", "truck", "motorbike", "train", "aeroplane"};  % Define the classes for detection
testData = yoloformat2trainingData(folderPath,classes,false);

save('testData_tal.mat','testData');

%% prepare data before test
load("Data/testData_tal.mat");                               % load data

imds = imageDatastore(testData.imagesPath);                 % path to image data store
blds = boxLabelDatastore(testData(:,2:end));                % all rows, and 2-end
dsVal   = combine(imds, blds);                            


len = length(imds.Files);                                   % length of files
classes = string(testData.Properties.VariableNames(2:end)); % All classes
detector = yolov4ObjectDetector("csp-darknet53-coco");
%% Test detector


% prepare a table with 3 columns - bbox, scores, labels
pred = table('Size',[len 3], ...
    'VariableTypes',{'cell','cell','cell'}, ...
    'VariableNames',{'Boxes','Scores','Labels'});

dt = zeros([len 1]);
reverseStr = '';

for i=1:len
    I = readimage(imds, i); % read image
    tic
    [bboxes, scores, labels] = detect(detector,I,Threshold=0.5); % detect classes in image
    dt(i) = toc;
    idx = ismember(string(labels)', classes);
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

% [cm, confClassNames] = confusionMatrix(metrics);
% chart = confusionchart(cm{:},confClassNames);
% title('Object Detection Confusion Matrix');



%% F1 score caculate

% 1. Extract precision and recall (these will be cell arrays)
[recall, precision, ~] = precisionRecall(metrics);

% 2. Calculate F1-score for each cell using cellfun
% This handles the element-wise math for every class automatically
f1Scores = cellfun(@(p, r) (2 .* p .* r) ./ (p + r + eps), ...
                   precision, recall, 'UniformOutput', false);

% 3. Extract the maximum F1-score from each class
maxF1PerClass = cellfun(@max, f1Scores);

% 4. Display the results clearly
classNames = metrics.ClassNames;

% for i = 1:numel(classNames)
%     fprintf('Class: %-15s | Max F1-Score: %.4f\n', classNames{i}, maxF1PerClass(i));
% end

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
yolov4 = struct('Detector', detector, ...
                'Metrics', metrics, ...
                'Predictions', pred, ...
                'f1Scores',f1, ...
                'AvgProcessTime',mean(dt));

save('yolov4_results.mat', 'yolov4');