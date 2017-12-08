clear all
close all
clc
trainingFeatures = [];
trainingLabels   = [];
load('filenames.mat');
load('test2ImNames.mat');
no_train_images=100;
no_test_images=25;

for i=1:25
    for j = 1:no_train_images
        image=imread(strcat('.',trainImNames{i,j}));
        image=imresize(image,[200 200]);
        if(size(image,3)==3)
            im1 = im2single(rgb2gray(image));
        else
            im1 = im2single(image);
        end
        [hog vis]= extractHOGFeatures(im1,'CellSize',[32 32]);
        trainingFeatures=[trainingFeatures;hog];
    end
    labels = repmat(i,no_train_images, 1);
    trainingLabels=[trainingLabels; labels];
end

classifier = fitcecoc(trainingFeatures, trainingLabels,'Coding','onevsone');

testFeatures = [];
testLabels   = [];

for u=1:25
    for v = 1:no_test_images
        image=imread(strcat('.',test2ImNames{u,v}));
        image=imresize(image,[200 200]);
        if(size(image,3)==3)
            im_test = im2single(rgb2gray(image));
        else
            im_test = im2single(image);
        end
        [hog vis]= extractHOGFeatures(im_test,'CellSize',[32 32]);
        testFeatures=[testFeatures;hog];
    end
    labels = repmat(u,no_test_images, 1);
    testLabels=[testLabels;labels];
end
        
predictedLabels = predict(classifier, testFeatures);

confMat = confusionmat(testLabels, predictedLabels);

res=confMat/25;
mean(diag(res))
