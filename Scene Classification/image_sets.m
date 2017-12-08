clear all
close all
clc
load('filenames.mat');
for i=1:1:25
    for j=1:1:100
        image=imread(strcat('.',trainImNames{i,j}));
        imwrite(image,strcat('C:\Users\user1\Documents\CS 381V A2\project3\cs381V.grauman\SUN397\training\',num2str(i),'\',num2str(j),'.jpg'));
    end
end

for u=1:1:25
    for v=1:1:25
        image=imread(strcat('.',test1ImNames{u,v}));
        imwrite(image,strcat('C:\Users\user1\Documents\CS 381V A2\project3\cs381V.grauman\SUN397\testing\',num2str(u),'\',num2str(v),'.jpg'));
    end
end


