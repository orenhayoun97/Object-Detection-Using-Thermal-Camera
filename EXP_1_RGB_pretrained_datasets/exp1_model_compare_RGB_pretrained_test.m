clc; clear; close all

% load data
load('detectors/yolov2_results.mat');
load('detectors/yolov3_results.mat');
load('detectors/yolov4_results.mat');
load('detectors/yolox_results.mat');
% clearvars -except metrics detector pred blds imds names
%%
deepNetworkDesigner(detector.Network)
analyzeNetwork(yolov3_detector.Network)

lgraph = yoloxLayers([640 640 3], 80, 'ModelName', 'large-coco');
analyzeNetwork(lgraph)
%%
yolov2 = {yolov2_detector; yolov2_metrics; pred_2; pred_f1_th{1}; {dt(1,:)}};
yolov3 = {yolov3_detector; yolov3_metrics; pred_3; pred_f1_th{2}; {dt(2,:)}};
yolov4 = {yolov4_detector; yolov4_metrics; pred_4; pred_f1_th{3}; {dt(3,:)}};
yolox  = {yolovx_detector ; yolovx_metrics ; pred_x; pred_f1_th{4}; {dt(4,:)}};

yolo = [yolov2 , yolov3, yolov4, yolox];
data = [imds, blds, names];

save("data","data");
save("yolo","yolo");
%%
load("yolo.mat");
load("data.mat");

% data
imds = data{1};
blds = data{2};
names = data{3};

% detectors
yolov2 = yolo{1};
yolov3 = yolo{2};
yolov4 = yolo{3};
yolox = yolo{4};

detector = {yolov2{1}, yolov3{1}, yolov4{1}, yolox{1}};
metrics = {yolov2{2}, yolov3{2}, yolov3{2}, yolox{2}};
pred = {yolov2{3}.pred, yolov3{3}.pred, yolov4{3}.pred, yolox{3}.pred};
pred_f1 = {yolov2{4}, yolov3{4}, yolov4{4}, yolox{4}};
dt = {yolov2{5}, yolov3{5}, yolov4{5}, yolox{5}};

clearvars -except metrics detector pred blds imds names pred_f1 dt metrics_f1

%% recall vs. 
len= length(detector);
figure;
t = tiledlayout(2, 2);
targetTh = 0.5;
f1_th = [0.59, 0.95, 0.75, 0.77];
for i=1:len
    nexttile;precision
    [recall,precision,scores] = precisionRecall(metrics_f1{i});
    ap = averagePrecision(metrics_f1{i});
    plot(recall{1}, precision{1}, 'LineWidth', 2,'DisplayName',names(i));
    title(sprintf('%s with ap=%f',names(i),ap));
    xlabel('recall');
    ylabel('precision');
    legend


    % finding the closest index to max threshold
    [~, idx] = min(abs(scores{1} - f1_th(i)));
    
    % get the values in this index
    precisionAtTh = precision{1}(idx);
    recallAtTh = recall{1}(idx);


    fprintf('Precision: %.2f, Recall: %.2f\n', precisionAtTh, recallAtTh);
    fprintf('AP is: %.2f \n', ap);
end

% exportgraphics(gcf, 'YOLO_f1_performance.png', 'Resolution', 300);

clearvars -except metrics detector pred blds imds names pred_f1 dt metrics_f1
%% Image test


[r,~] = find(cell2mat(metrics{4}.ImageMetrics.AP) < 0.8); % num->r(num)
len = length(r);
imgNum = randperm(len, 4);
figure;
t = tiledlayout(4, 5);
t.TileSpacing = 'compact';
t.Padding = 'compact';
for i=1:length(imgNum)
    num = imgNum(i);
    I = readimage(imds,num);
    
    % Display the first image with detected bounding boxes
    % LabelScoreStr = compose("%s-%.2f",pred.Labels{num},pred.Scores{num});
    for ii=1:length(detector)
        nexttile; % subplot(4,5,(i-1)*5+ii);
        detectedImage = insertObjectAnnotation(I, 'rectangle', pred{ii}.Boxes{num}, "","LineWidth",4,'FontSize',25);
        imshow(detectedImage);
        title(sprintf('%s image =%d',names(ii),num));
    end
    nexttile; % subplot(4,5,(i-1)*5+ii+1);
    groundTruthImage = insertObjectAnnotation(I, 'rectangle', blds.LabelData{num,1},"","LineWidth",4,'FontSize',25);
    imshow(groundTruthImage);
    title(sprintf('GroundTruth image %d',num));
    
end 

% time_table = cell2table(dt,"RowNames",names,"VariableNames",["img 1","img 2", "img3", "img 4", "av"]);
% exportgraphics(gcf, 'YOLO_Results.png', 'Resolution', 300);

clearvars -except metrics detector pred blds imds names pred_f1 dt metrics_f1
%%
path = '../../dataset/RGB data set/RGB-Person-OREN-valid-dataset';
classes = {"person"};
A = yoloformat2trainingData(path, classes, false);
%%
imds = imageDatastore(A.imagesPath);
%% Timing test

% figure;
% t = tiledlayout(4, 5);
% t.TileSpacing = 'compact';
% t.Padding = 'compact';
reverseStr = '';
j=1;
len = length(blds.LabelData);
dt = zeros([length(detector),len]);
objNum = zeros([length(detector),len]);
for i=1:length(detector)
    for ii=1:len
        I = readimage(imds,ii);
        tic
        [~, ~, labels] = detect(detector{i},I,Threshold=0.5); % detect classes in image
        dt(i,ii) = toc;

        msg = sprintf('detector=%s\nProgress: %.2f%%',names(i), ((j)/(len*4))*100);
        fprintf(reverseStr);                                % delete line
        fprintf('%s',msg);                                  % write msg
        reverseStr = repmat('\b', 1, length(msg));
        j=j+1;
    end

end 
for i=1:length(detector)
    fprintf("\n%s status:\nmu=%.3f [sec]\nAvg frame rate: %.3f [Hz]\n",names(i),mean(dt(i,:)),1/mean(dt(i,:)));
end
% save("yoloTiming","dt");

clearvars -except metrics detector pred blds imds names pred_f1 dt metrics_f1
%% get metrics
load("pred_F1th.mat");
for i=1:length(detector)
    metrics_f1{i} = evaluateObjectDetection(pred_f1_th{i}, blds);
end
%% F1 score

len= length(detector);
figure;
t = tiledlayout(4, 2);
t.TileSpacing = 'compact';
t.Padding = 'compact';
disp("******** START F1 SCORE TEST  ********");
for i=1:len
    nexttile;
    [recall, precision, scores] = precisionRecall(metrics_f1{i});
    ap = averagePrecision(metrics_f1{i});
    f1_scores = (2 .* precision{1} .* recall{1}) ./ (precision{1} + recall{1});
    f1_scores(isnan(f1_scores)) = 0;
    [maxF1, maxIdx] = max(f1_scores);
    optimalThreshold = scores{1}(maxIdx);
    plot(recall{1}, precision{1}, 'LineWidth', 2);
    grid on;
    xlabel('Recall');
    ylabel('Precision');
    title('PR Curve of ' + names(i) + ' (AP: '+ num2str(ap, 2) + ')');


    nexttile;
    plot(scores{1}, f1_scores, 'r', 'LineWidth', 2);
    hold on;
    plot(optimalThreshold, maxF1, 'ko', 'MarkerSize', 10, 'LineWidth', 2);
    grid on;
    xlabel('Confidence Threshold');
    ylabel('F1 Score');
    title(names(i) + ':Optimal Threshold: ' + num2str(optimalThreshold, 2));
    legend('F1 Curve', 'Max F1 Point',"Location","southwest");
    fprintf('\n** %s status: **\n',names(i));
    fprintf('Opitmal threshold is: %.2f \n', optimalThreshold);
end
% exportgraphics(gcf, 'YOLO_F1th_score.png', 'Resolution', 300);

clearvars -except metrics detector pred blds imds names pred_f1 dt metrics_f1
%% detector f1 threshold

pred = table('Size',[len 3], ...
    'VariableTypes',{'cell','cell','cell'}, ...
    'VariableNames',{'Boxes','Scores','Labels'});
% Variables
reverseStr = '';
classNames = "person";
j=1;
f1_th = [0.59, 0.95, 0.75, 0.77];
len = length(blds.LabelData);
for i=1:length(detector)
    for ii=1:len
        I = readimage(imds,ii);
        [bboxes, scores, labels] = detect(detector{i},I,Threshold=f1_th(i)); % detect classes in image
        idx = (labels == 'person');
        pred.Boxes{ii} = bboxes(idx,:);
        pred.Scores{ii} = scores(idx,:);
        pred.Labels{ii} = categorical(labels(idx,:), classNames);

        msg = sprintf('\ndetector=%s\nProgress: %.2f%%',names(i), ((j)/(len*4))*100);
        fprintf(reverseStr);                                % delete line
        fprintf('%s',msg);                                  % write msg
        reverseStr = repmat('\b', 1, length(msg));
        j=j+1;
    end
    pred_f1_th{i} = pred;
end

save("pred_F1th","pred_f1_th");
clearvars -except metrics detector pred blds imds names pred_f1 dt metrics_f1
%% conufusion matrix
T = table('Size',[2 2], ...
    'VariableTypes',{'int64','int64'}, ...
    'VariableNames',{'person','unnamed'}, ...
    'RowNames',{'person','unnamed'});

for i=1:length(detector)
    disp(names(i));
    [cm,confusionClassNames] = confusionMatrix(metrics{i});
    % T.Properties.VariableNames = confusionClassNames{:};
    % T.Properties.RowNames = confusionClassNames{:};
    T.Variables = cm{:};
    all_cm{i} = T;
    disp(T);
end

clearvars -except metrics detector pred blds imds names pred_f1 dt metrics_f1

%%

%% הפקת טבלת סיכום עם אחוזי שיפור (Optimal vs. Baseline 0.5)
numModels = length(metrics);

PerformanceSummary = table('Size',[numModels 7], ...
    'VariableTypes',{'string','double','double','double','double','double','string'}, ...
    'VariableNames',{'Model','Th_Optimal','Precision_0_5','Precision_Opt','Recall_0_5','Recall_Opt','Precision_Improvement'});

for i = 1:numModels
    [recall, precision, scores] = precisionRecall(metrics{i});
    rec = recall{1}; prec = precision{1}; sc = scores{1};
    
    % 1. מציאת נקודת ה-F1 המקסימלית (Optimal)
    f1 = (2 .* prec .* rec) ./ (prec + rec);
    f1(isnan(f1)) = 0;
    [~, optIdx] = max(f1);
    
    % 2. מציאת הנקודה הכי קרובה לסף 0.5 (Baseline)
    [~, baseIdx] = min(abs(sc - 0.5));
    
    % 3. חישוב אחוז השיפור ב-Precision
    p_base = prec(baseIdx);
    p_opt = prec(optIdx);
    improvement = ((p_opt - p_base) / p_base) * 100;
    
    % מילוי הטבלה
    PerformanceSummary.Model(i) = names(i);
    PerformanceSummary.Th_Optimal(i) = sc(optIdx);
    PerformanceSummary.Precision_0_5(i) = p_base;
    PerformanceSummary.Precision_Opt(i) = p_opt;
    PerformanceSummary.Recall_0_5(i) = rec(baseIdx);
    PerformanceSummary.Recall_Opt(i) = rec(optIdx);
    PerformanceSummary.Precision_Improvement(i) = sprintf('%.1f%%', improvement);
end

disp(PerformanceSummary);



%%

[precision,recall,scores] = precisionRecall(metrics{1});


targetTh = 0.5;

% מציאת האינדקס של הסף הקרוב ביותר בוקטור ה-scores
[~, idx] = min(abs(scores{1} - targetTh));

% שליפת הערכים המדויקים באותו אינדקס
precisionAtTh = precision{1}(idx);
recallAtTh = recall{1}(idx);

fprintf('בסביבת סף %.2f:\n', targetTh);
fprintf('Precision: %f, Recall: %f\n', precisionAtTh, recallAtTh);