clc; close all; clear;

%% 1. Directory and Classes Setup
HIVdir = '../../dataset/HIT-UAV-drone';
classes = ["person", "car", "motorcycle", "overvehicle", "don't care", "drone"];

%% 2. Load Data (Train & Validation)
load('MATLAB data/trainingData.mat');
load('MATLAB data/valData.mat');

load("MATLAB data/testData_tal.mat")
load("MATLAB data/myYOLOv4_Model_1.mat");

%% 3. Create Datastores
% Train Datastore
imdsTrain = imageDatastore(trainingData.imagesPath);
bldsTrain = boxLabelDatastore(trainingData(:,2:end));
dsTrain   = combine(imdsTrain, bldsTrain);

% Validation Datastore
imdsVal = imageDatastore(valData.imagesPath);
bldsVal = boxLabelDatastore(valData(:,2:end));
dsVal   = combine(imdsVal, bldsVal);

%% draft 


%% 4. Input Size 
inputSize = [320 320 3]; 


trainingDataForEstimation = transform(dsTrain, @(data)preprocessData(data, inputSize));

numAnchors = 6; 
disp('Estimating Anchor Boxes...');
[anchors, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors);


area = anchors(:,1).*anchors(:,2);
[~,idx] = sort(area,"descend");
anchors = anchors(idx,:);
anchorBoxes = {anchors(1:2,:) ; anchors(3:4,:); anchors(5:6,:)};

disp(['Mean IoU of estimated anchors: ', num2str(meanIoU)]);

%% 5. Configure YOLOv4 Network
detector = yolov4ObjectDetector("csp-darknet53-coco", classes, anchorBoxes, InputSize=inputSize);
%detector = yolov4ObjectDetector("", classes, anchorBoxes, InputSize=inputSize);
%% 6. Training Options

options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...         
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 10, ...         
    'MaxEpochs', 40, ...                  
    'MiniBatchSize', 8, ...               
    'L2Regularization', 0.0005, ...
    'ValidationData', dsVal, ...           
    'ValidationFrequency', 50, ...
    'OutputNetwork', 'best-validation-loss', ... 
    'Verbose', true, ...
    'ResetInputNormalization', false, ...
    'ExecutionEnvironment', 'gpu', ...
    'Plots', 'training-progress');

% 7. Train the Model
disp('Starting Training...');
[trainedDetector, info] = trainYOLOv4ObjectDetector(dsTrain, detector, options);

% שמירת המודל המאומן
save('myYOLOv4_Model_1.mat', 'trainedDetector', 'info');
disp('Model saved successfully!');

%% 8. Test on a Random Image (Inference)
folderPathImageSource = fullfile(HIVdir,'/test/images'); 
filePattern = fullfile(folderPathImageSource, '*.jpg'); 
filesImages = dir(filePattern);

if ~isempty(filesImages)
    randNum = randi([1 length(filesImages)]);
    I = imread(fullfile(folderPathImageSource, filesImages(randNum).name));
    
    
    [bboxes, scores, labels] = detect(trainedDetector, I, Threshold=0.3);
    
 
    if ~isempty(bboxes)
        detectedImg = insertObjectAnnotation(I, "Rectangle", bboxes, labels, "LineWidth", 3, 'FontSize', 14);
        figure; imshow(detectedImg); title('YOLOv4 Detection Results');
    else
        figure; imshow(I); title('No objects detected');
    end
else
    disp('No test images found in the specified directory.');
end

%% Helper Function for Resizing Data (Required for estimateAnchorBoxes)
function data = preprocessData(data, targetSize)
    
    for ii = 1:size(data,1)
        I = data{ii,1};
        bboxes = data{ii,2};
        
        sz = size(I);
        scale = targetSize(1:2) ./ sz(1:2);
        
        I = imresize(I, targetSize(1:2));
        bboxes = bboxresize(bboxes, scale);
        
        data{ii,1} = I;
        data{ii,2} = bboxes;
    end
end


%% Test detector
warning('off', 'MATLAB:imagesci:png:libraryWarning');
len = height(testData);
imds = imageDatastore(testData.imagesPath);

detector=trainedDetector;
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

%%  Get metrics
load("MATLAB data/testData_tal.mat")
blds = boxLabelDatastore(testData(:,2:end));
metrics = evaluateObjectDetection(pred, blds);

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

%% Save the YOLOv4 detector, metrics, and predictions into a structure
yolov4 = struct('Detector', detector, ...
                'Metrics', metrics, ...
                'Predictions', pred, ...
                'f1Scores',f1, ...
                'AvgProcessTime',mean(dt));

save('yolov4_results.mat', 'yolov4');