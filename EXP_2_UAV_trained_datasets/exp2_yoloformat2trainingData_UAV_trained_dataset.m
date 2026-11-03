function trainingData = yoloformat2trainingData(folderPath, classes, saveData)
% This function convert yolo format to MATLAB format
% # FolderPath is the path to images and labels folder
%   labels format is text
%   images format is jpg type
% # classes - cell object the obtain classes
%   exp: {"person", "car", "animal"}
%        {    0   ,   1  ,    2    }
%        0->person, 1->car, 2->animal

% Syntax:
%   trainData = yoloformat2trainingData(folderPath,save)
    
    arguments
        folderPath (1,1) string
        classes
        saveData (1,1) logical = 1
    end
    
    % Explore the folder

    % folderPath = 'train';
    imagesFiles = dir(fullfile(folderPath,'/images'));
    labelsFiles = dir(fullfile(folderPath,'/labels'));
    
    % Check if path found
    if isempty(imagesFiles) || isempty(labelsFiles)
        disp('[ERROR] Check file path!!');
        return;
    end
    
    % Remove all dirs from struct
    isNotFolder = ~[imagesFiles.isdir]; 
    imagesFiles = imagesFiles(isNotFolder);
    
    isNotFolder = ~[labelsFiles.isdir];
    labelsFiles = labelsFiles(isNotFolder);
    
    % Check if the length of files are equal
    len = length(imagesFiles);
    if length(imagesFiles) ~= length(labelsFiles)
        disp("[ERROR] length of images and labels are different");
        return;
    end
    % Variable used
    reverseStr = '';
    
    % main loop
    for i = 1:len
        imagesPath{i,1} = fullfile(imagesFiles(i).folder,imagesFiles(i).name);
        I = imread(imagesPath{i,1});
        
        bboxYoloFormat = load(fullfile(labelsFiles(i).folder,labelsFiles(i).name));
        temp_bbox = ConvertYoloFormat2MatlabFormat(bboxYoloFormat, size(I));
        if isempty(temp_bbox)                               % skip if there is no objects in image
            continue;
        end
        uniqueObj = unique(temp_bbox(:,1));                 % check for unique object
        for j=1:length(uniqueObj)
            [r,~] = find(uniqueObj(j)==temp_bbox(:,1));     % get rows of object 'j'         
            temp = temp_bbox(r',:);                         % get bbox of object 'j' 
            temp(:,1) = [];                                 % remove object column
            % exp of object : [0, 1 , 4, 6] etc
            bbox{i,uniqueObj(j)+1} = temp;                  % save bbox for each object in image 'i'  
        end

        % Process time update
        msg = sprintf('Progress: %.2f%%', (i/len)*100);
        fprintf(reverseStr);                                % delete line
        fprintf('%s',msg);                                  % write msg
        reverseStr = repmat('\b', 1, length(msg));          % prepare delte for next intration
    end

    td = [imagesPath,bbox];                                 % combine info
    trainingData = cell2table(td,"VariableNames",["imagesPath",classes]);                          % change to table
    
    % if save true - save the data to workspace
    if saveData
        save("trainingData", "trainingData");
    end
    
    fprintf("\nDone to convert!!!!\n");

    function bbox_pos = ConvertYoloFormat2MatlabFormat(bbox_norm,img_sz)
    % Change YOLO format to MATLAB format for bounding box
    % [x_center y_center width hight]_normilize -> [x y width hight]
    % input: bbox_norm=[ID x_normalize_center y_normalize_center width_normalize
    % high_normlize], image_size[row col dem]
    % output: bbox_pos = [x y width high]
    
        if isempty(bbox_norm)
            bbox_pos = bbox_norm;
            return; 
        end
        new_sz = fliplr(img_sz(1:2));
        centerPoint = round(bbox_norm(:,2:3).*new_sz);
        w_h = round(bbox_norm(:,4:5).*new_sz);                       % [width hight]
        leftCorner = round(centerPoint - [w_h(:,1)/2 w_h(:,2)/2]);   % leftCorner
        bbox_pos = [bbox_norm(:,1) leftCorner w_h(:,1) w_h(:,2)];
    end
    
end