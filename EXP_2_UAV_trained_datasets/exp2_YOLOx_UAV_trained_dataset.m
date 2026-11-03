clc;close all; clear


% HIV-UAV directory
HIVdir = '../../dataset/HIT-UAV-drone';
classes = {"person", "car", "motorcycle", "overvehicle", "don't care", "drone"};

%% Create a DATA
% Create a traning data from yolo format
warning('off', 'MATLAB:imagesci:png:libraryWarning');
folderPath = fullfile(HIVdir,'/train');
trainingData = yoloformat2trainingData(folderPath,classes,false);

% Create a validation data from yolo format
warning('off', 'MATLAB:imagesci:png:libraryWarning');
folderPath = fullfile(HIVdir,'/val');
valData = yoloformat2trainingData(folderPath,classes,false);

% Create a testing data from yolo format
warning('off', 'MATLAB:imagesci:png:libraryWarning');
folderPath = fullfile(HIVdir,'/test');
testData = yoloformat2trainingData(folderPath,classes,false);

save('trainingData_mac','trainingData');
save('testData_mac','testData');
save('valData_mac','valData');

%% Custom Training
% Train
imds = imageDatastore(trainingData.imagesPath);  % training images
blds = boxLabelDatastore(trainingData(:,2:end)); % training bbox
dsTrain = combine(imds,blds);

% Validation
imds = imageDatastore(valData.imagesPath);       % val images
blds = boxLabelDatastore(valData(:,2:end));      % val bbox
dsVal = combine(imds,blds);


% Choose a input
targetSize = [320 320 3];
sz = size(readimage(imds,1));
scale = sz(1)./sz(2);
if scale>1 % row is bigger
    inputImage = [fix(targetSize(1)*scale/32)*32 targetSize(2:3)];
else
    inputImage = [targetSize(1) fix(targetSize(2)*scale/32)*32 targetSize(3)];
end


clear imds blds trainingData valData testData
classes = ["person", "car", "motorcycle", "don't care1", "don't care2", "drone"];
detector = yoloxObjectDetector("tiny-coco", classes, InputSize=inputImage);

%%%%%%%% CAN CHANGE %%%%%%%%%%%
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.01, ...          % Start high for Adam
    'LearnRateSchedule', 'piecewise', ...  % Drop rate over time
    'LearnRateDropFactor', 0.1, ...        % Reduce by 10x
    'LearnRateDropPeriod', 5, ...          % Every 20 epochs
    'MaxEpochs', 10, ...                   % Adjust based on data size
    'MiniBatchSize', 32, ...               % Increase if your GPU memory allows
    'L2Regularization', 0.0005, ...        % Prevents overfitting
    'ValidationData', dsVal, ...           % CRITICAL: Use a separate validation set
    'ValidationFrequency', 20, ...        % Check mAP/Loss every epoch ,in our case 100 iterations
    'OutputNetwork', 'best-validation-loss', ... % Keep the BEST model, not the LAST
    'Verbose', true, ...
    'DispatchInBackground', true', ...     % This uses CPU workers to load data
    'Plots', 'training-progress', ...      % Visual monitor
    'ExecutionEnvironment', 'gpu',...      % Use 'auto' or 'cpu' if no GPU
    'WorkerLoad', 1);                      % Uses 100% of available workers for loading 

[detector, info] = trainYOLOXObjectDetector(dsTrain, detector, options);

save('myModel_win_2.mat', 'detector');

%% Test detector
load("myModel_win_2.mat")
load('MATLAB data/testData.mat');
imds = imageDatastore(testData.imagesPath);       % test images


classes = string(testData.Properties.VariableNames);
classes = classes(2:end);
len = length(imds.Files);
pred = table('Size',[len 3], ...
    'VariableTypes',{'cell','cell','cell'}, ...
    'VariableNames',{'Boxes','Scores','Labels'});

dt = zeros([len 1]);
reverseStr = '';
warning('off', 'MATLAB:imagesci:png:libraryWarning');
for i=1:len
    I = readimage(imds, i); % read image
    tic
    [bboxes, scores, labels] = detect(detector,I,Threshold=0.5); % detect classes in image
    dt(i) = toc;
    idx = ismember(string(labels)', classes); 
    pred.Boxes{i} = bboxes(idx,:);
    pred.Scores{i} = scores(idx,:);
    pred.Labels{i} = categorical(labels(idx,:), classes);

    % Process time
    msg = sprintf('Progress: %.2f%%', (i/len)*100);
    fprintf(reverseStr);                                % delete line
    fprintf('%s',msg);                                  % write msg
    reverseStr = repmat('\b', 1, length(msg));          % prepare delte for next intration
end

%% Get metrics
blds = boxLabelDatastore(testData(:,2:end));      % test bbox
metrics = evaluateObjectDetection(pred, blds);

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

%%

yolox = struct('Detector', detector, ...
                'Metrics', metrics, ...
                'Predictions', pred, ...
                'f1Scores',f1, ...
                'AvgProcessTime',mean(dt));

save('yolox_results.mat', 'yolox');
