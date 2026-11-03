function bbox_pos = ConvertYoloFormat2MatlabFormat(bbox_norm,img_sz)
% Change YOLO format to MATLAB format for bounding box
% [x_center y_center width hight]_normilize -> [x y width hight]
% input: bbox_norm[ID x_normalize_center y_normalize_center width_normalize
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