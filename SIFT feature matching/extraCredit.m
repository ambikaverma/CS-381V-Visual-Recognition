clear all
close all
clc
% VLFeat is added to path to allow running vl_sift function
run('C:\Users\user1\Downloads\vlfeat-0.9.20-bin.tar\vlfeat-0.9.20-bin\vlfeat-0.9.20\toolbox\vl_setup');

%template image of the object to be detected in various scenes
templatename = 'christ_church_obj.jpg';

%scene images in which the template object is detected
scenenames = {'christ_church_test2.jpg','christ_church_test9,jpg','christ_church_test1,jpg','christ_church_test5.jpg','christ_church_test6.jpg','christ_church_test4.jpg','test10.jpg', 'test11.jpg', 'test12.jpg'};


%read the template image, convert image from RGB to grayscale followed by
%conversion to single datatype
im1 = im2single(rgb2gray(imread(templatename)));

%show template image
figure
imshow(im1)
title('Object Template Image')

% Extract SIFT features from the template image.
%
% 'f' refers to a matrix of "frames".  It is 4 x n, where n is the number
% of SIFT features detected.  Thus, each column refers to one SIFT descriptor.  
% The first row gives the x positions, second row gives the y positions, 
% third row gives the scales, fourth row gives the orientations.  You will
% need the x and y positions for this assignment.
%
% 'd' refers to a matrix of "descriptors".  It is 128 x n.  Each column 
% is a 128-dimensional SIFT descriptor.
%
% See VLFeats for more details on the contents of the frames and
% descriptors.
[f1, d1] = vl_sift(im1);

% count number of descriptors found in im1
n1 = size(d1,2);

% Loop through the scene images
for scenenum = 1:length(scenenames)
    n1 = size(d1,2);
im2 = im2single(rgb2gray(imread(scenenames{scenenum})));

    %show scene image currently being worked on
    figure
    imshow(im2)
    title(['Test Scene Image' num2str(scenenum)])
    
    % Extract SIFT features from this scene image
    [f2, d2] = vl_sift(im2);
    n2 = size(d2,2);
    
    % Compute the Euclidean distance between that descriptor
    % and all descriptors in im2
    dists = dist2(double(d1)', double(d2)');
    
    % Sort those distances
    [sortedDists, sortedIndices] = sort(dists,2, 'ascend');
    
    %sortedDists_min contains the best match (minimum distance) for each descriptor of im1 in
    %im2
    sortedDists_min=sortedDists(:,1);
    
    %sortedDists_top2 contains the two best matches for each descriptor of
    %im1 in im2 
    %this is useful while performing Lowe's ratio test
    sortedDists_top2=sortedDists(:,1:2);
    
    % Take the first neighbor as a candidate match.
    % Record the match as a column in the matrix 'matchMatrix',
    % where the first row gives the index of the feature from the first
    % image, the second row gives the index of the feature matched to it in
    % the second image, and the third row records the distance between
    % them.
    matchMatrix = [(1:n1); sortedIndices(:,1)'; sortedDists_min'];
    
    figure
    showLinesBetweenMatches(im1, im2, f1, f2, matchMatrix);
    title(['All SIFT Matches initially obtained (' num2str(n1) 'matches)'])
    
    % Thresholded nearest neighbors
    % A threshold of 0.8*(mean distance of nearest neighbor) is applied on
    % the raw euclidean distances computed
    mean_num=mean(sortedDists_min);
    threshold=0.8*mean_num;
    
    % filtered_feat_indices keeps track of the descriptors which pass the
    % nearest neighbor threshold filtering 
    filtered_feat_indices=[];
    
    % top2_dists is the new set of distances of the two nearest descriptors
    % from im2 for each descriptor of im1
    top2_dists=[];
    
    % iterate through each descriptor, n1 is total number of descriptors in
    % im1
    for i=1:n1
        
        % Threshold condition
        if (sortedDists_min(i)<threshold)
            filtered_feat_indices=[filtered_feat_indices;i];
            top2_dists=[top2_dists ;sortedDists_top2(i,:)];
        end
    end
    
    % matchMatrix is updated to contain only the descriptors of im1 and
    % their matched counterparts in im2 (along with the distance) which
    % satisfy the nearest neighbor threshold
    matchMatrix=matchMatrix(:,filtered_feat_indices);
    
    % n1 now reflects the number of decriptors of im1 after applying
    % nearest neighbor threshold
    n2=length(filtered_feat_indices);
    
    figure
    showLinesBetweenMatches(im1, im2, f1, f2, matchMatrix);
    title(['SIFT Matches after applying raw threshold (' num2str(n2) 'matches)'])

    % Lowe's ratio test or Thresholded ratio test
    % filtered_feat_indices2 keeps track of the descriptors which pass the
    % thresholded ratio test 
    filtered_feat_indices2=[];
    
    % iterate through each descriptor in im1
    for u=1:1:n2
        % ratio of distance to first nearest neighbor vs distance to second
        % nearest neighbor is computed
        % Threshold ratio = 0.6
        ratio(u)=top2_dists(u,1)/top2_dists(u,2);
        if ratio(u)<0.6
            filtered_feat_indices2=[filtered_feat_indices2; u];
        end
    end

    % matchMatrix is updated to contain only the descriptors of im1 and
    % their matched counterparts in im2 (along with the distance) which
    % pass the thresholded ratio test
    matchMatrix=matchMatrix(:,filtered_feat_indices2);
    
    % n1 now reflects the number of decriptors of im1 after applying
    % Lowe's ratio test
    n3=length(filtered_feat_indices2);
    
    figure
    showLinesBetweenMatches(im1, im2, f1, f2, matchMatrix);
    title(['SIFT Matches after applying Lowe ratio test (' num2str(n3) 'matches)'])
    
    % RANSAC Algorithm
    
    % max stores the largest number of inliers obtained till present
    % iteration
    max=0;
    
    % 100 iterations are executed
    for iter=1:100
        try
            % randperm chooses a set of three values each drawn randomly
            % from the set containing values 1 to No. of descriptors
            % inclusive
            random_indexes=randperm(size(matchMatrix,2),3);
            
            % random_indexes are used to obtain the corresponding
            % descriptors of im1 from 1st row of matchMatrix
            % The corresponding descriptor indexes are then used to obtain
            % the x and y positions from the first two rows of f1
            temp_locs=f1(1:2,matchMatrix(1,random_indexes))';
            
            % random_indexes are used to obtain the corresponding
            % descriptors of im2 from 2nd row of matchMatrix
            % The corresponding descriptor indexes are then used to obtain
            % the x and y positions from the first two rows of f2
            scene_locs=f2(1:2,matchMatrix(2,random_indexes))';
            
            % MATLAB function cp2tform is used to solve for the 6 affine
            % transformation parameters given the 3 random points
            % The result obtained is saved in tform_test
            tform_test=cp2tform(temp_locs,scene_locs,'affine');
            
            % the obtained affine transformation matrix is then applied for
            % all descriptors in im1
            % resulting in a set of x,y transformed positions which are
            % then compared to the x,y positions of descriptors in im2 to
            % determine the number of inliers
            trans_temp_coords = tformfwd(tform_test, f1(1:2,matchMatrix(1,:))');
            scene_coords=f2(1:2,matchMatrix(2,:))';
%              x_coordinates_to_be_transformed=f1(1,matchMatrix(1,:));
%              y_coordinates_to_be_transformed=f1(2,matchMatrix(1,:));
%              x_coordinates_second_image_original=f2(1,matchMatrix(2,:));
%              y_coordinates_second_image_original=f2(2,matchMatrix(2,:));
%              [u ,v]=tformfwd(tform_test,x_coordinates_to_be_transformed,y_coordinates_to_be_transformed);
%              len=size(x_coordinates_second_image_original,2);
%           len=size(scene_coords,1);
%           count=0;
%           for i=1:len
% %              distance=sqrt((trans_temp_coords(i,1)-scene_coords(i,1))^2+(trans_temp_coords(i,2)-scene_coords(i,2))^2);
%                  distance2(i)=sqrt((u(i)-x_coordinates_second_image_original(i))^2+(v(i)-y_coordinates_second_image_original(i))^2);
%         
%           if(distance2(i)<5)
%                       count=count+1;
%                   end
%           end
            % following formula for euclidean distance is used to determine
            % the number of inliers based on the transformed x,y positions
            % and available positions for im2
            distance=sqrt((trans_temp_coords(:,1)-scene_coords(:,1)).^2+(trans_temp_coords(:,2)-scene_coords(:,2)).^2);

            % following computation returns the number of inliers
            % where an inlier is defined as a set of transform x,y
            % positions and scene x,y positions which are at a distance
            % (less than the threshold distance specified) from each other. 
            no_of_inliers=numel(find(distance<1));
            inliers=find(distance<1);
            
            % for each iteration max contains the highest number of inliers
            % seen till that iteration
            % if the current no. of inliers is higher than max, the current
            % affine transformation matrix is saved as the best result
            if (no_of_inliers>max)
                max=no_of_inliers;
                best_transform=tform_test;
                final_inliers=inliers;
            end
        end
    end
    matchMatrix=matchMatrix(:,final_inliers);
    figure
    showLinesBetweenMatches(im1, im2, f1, f2, matchMatrix);
    title(['SIFT Matches after applying RANSAC (' num2str(max) 'matches)'])
    
    if (max>3)
    % points corresponds to the set of the four corners of the object
    % template image im1
    points=[1 1;size(im1,2) 1;1 size(im1,1);size(im1,2) size(im1,1)];
    
    % the transformation matrix obtained from RANSAC is used to transform
    % the corner points of im1 to the coordinates of im2
    points_f=tformfwd(best_transform,points);
    
    % drawRectangle subroitine is used to plot the bounding box, depicting
    % the area where object is detected.
    figure
    imshow(im2)
    drawRectangle(points_f','g');
    else
        fprintf('Object not detected in the scene image\n');
    end
end
