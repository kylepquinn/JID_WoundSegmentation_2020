% Quantitative Tissue Diagnostics Lab
% Jake Jones, Kyle Quinn
% University of Arkansas
% Submitted to Journal of Investigative Dermatology, June 2020
% for manuscript "Automated Quantitative Analysis of Wound Histology using
% Deep Learning Neural Networks"

close all;
clear;
clc;

%% Directory Navigation
shire='Input File Directory';
cd(shire)

%% Initialize network settings
imageSize = [512,512,3];
numClasses = 7;

% Custom encoder depth
encoderDepth = 4;
lgraph = unetLayers(imageSize,numClasses,'EncoderDepth',encoderDepth);

%% Loading data sets
dataSetDir = fullfile(shire);

% Training Set Image Data
imageDir = fullfile(dataSetDir,'H&E Image Directory');  
labelDir = fullfile(dataSetDir,'Segmented H&E Image Directory'); 

% Validation Set Image Data
imageDirVal = fullfile(dataSetDir,'H&E Image Directory');
labelDirVal = fullfile(dataSetDir,'Segmented H&E Image Directory');

%% Image datastore
imds = imageDatastore(imageDir);
imdv = imageDatastore(imageDirVal);

% Define class labels and associated IDs
classNames = ["epidermis","dermis","granulation","scab","follicle","background","muscle"];
labelIDs   = [36,72,108,144,180,216,252];

% Create a pixel datastore holding the ground truth pixel labels for the training images
pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);
pxdv = pixelLabelDatastore(labelDirVal,classNames,labelIDs);

% Data source for training semantic segmentation network
ds = pixelLabelImageDatastore(imds,pxds);
dv = pixelLabelImageDatastore(imdv,pxdv);

%% Training options
UNetHETrainOptions = trainingOptions('adam','InitialLearnRate',1e-3,'MaxEpochs',100,'VerboseFrequency',10,'ValidationData',dv,'ValidationFrequency',50,'Shuffle','every-epoch','MiniBatchSize',4);
% Validation frequency is every epoch
% Every epoch is shuffled
% Mini-batch size of 4

%% Train the network and keep diary
UNetHESegmentation = trainNetwork(ds,lgraph,UNetHETrainOptions);
disp('Network Training Complete.')

%% Save data
disp('Saving Network...')
save UNetHEAuto
save UNetHETrainOptionsAuto100
disp('Network saved.')

disp('Training has completed.')