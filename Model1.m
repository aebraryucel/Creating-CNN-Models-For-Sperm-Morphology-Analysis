

path={'E:\DATASETS\SCIAN\Set1\fold'};


for i=1:5
 
clear ournet;


char=int2str(i); %for döngüsünden 1 yerine i olcak.    
trainconcat=strcat(char,'\train');
trainpath=strcat(path,trainconcat);


imds = imageDatastore(trainpath,'IncludeSubfolders',true,'LabelSource','foldernames');


augimds=augmentedImageDatastore([131 131 3],imds,'ColorPreprocessing','gray2rgb');


testconcat=strcat(char,'\test');
testpath=strcat(path,testconcat);




testds=imageDatastore(testpath,'IncludeSubfolders',true,'LabelSource','foldernames');
augtest=augmentedImageDatastore([131 131 3],testds,'ColorPreprocessing','gray2rgb');



inputLayer=imageInputLayer([131 131 3]);

conv1=convolution2dLayer(3,32,'Padding',[0 0 0 0]);
relu1=reluLayer();

conv2=convolution2dLayer(3,32,'Padding',[0 0 0 0]);
relu2=reluLayer();
pool1=maxPooling2dLayer([3 3],'Stride',2,'Padding',[0 0 0 0]);

conv3=convolution2dLayer(3,64,'Padding',[0 0 0 0]);
relu3=reluLayer();

conv4=convolution2dLayer(3,64,'Padding',[0 0 0 0]);
relu4=reluLayer();
pool2=maxPooling2dLayer([3 3],'Stride',2,'Padding',[0 0 0 0]);

conv5=convolution2dLayer(3,64,'Padding',[0 0 0 0]);
relu5=reluLayer();

conv6=convolution2dLayer(3,128,'Padding',[0 0 0 0]);
relu6=reluLayer();

conv7=convolution2dLayer(3,128,'Padding',[0 0 0 0]);
relu7=reluLayer();
pool3=maxPooling2dLayer([3 3],'Stride',2,'Padding',[0 0 0 0]);

conv8=convolution2dLayer(3,128,'Padding',[0 0 0 0]);
relu8=reluLayer();
conv9=convolution2dLayer(3,128,'Padding',[0 0 0 0]);
relu9=reluLayer();

conv10=convolution2dLayer(3,128,'Padding',[0 0 0 0]);
relu10=reluLayer();
pool4=maxPooling2dLayer([3 3],'Stride',2,'Padding',[0 0 0 0]);



fc2=fullyConnectedLayer(1024);
relu12=reluLayer();

fc3=fullyConnectedLayer(5);
s3=softmaxLayer();
c3=classificationLayer();






layers=[inputLayer;conv1;relu1;conv2;relu2;pool1;conv3;relu3;conv4;relu4;pool2;conv5;relu5;conv6;relu6;conv7;relu7;pool3;conv8;relu8;conv9;relu9;conv10;relu10;pool4;fc2;relu12;fc3;s3;c3];


valFreq=floor(numel(augimds.Files)/16);

options=trainingOptions('sgdm','MaxEpochs',50,'InitialLearnRate',0.0001,'ValidationData',augtest,'ValidationFrequency',valFreq,...
    'Verbose',true,'Shuffle','every-epoch','Plots','training-progress','MiniBatchSize',16,'ExecutionEnvironment','gpu',...
    'LearnRateSchedule','piecewise','LearnRateDropFactor',0.1,'LearnRateDropPeriod',9999);

tic
ournet=trainNetwork(augimds,layers,options);  
toc
analyzeNetwork(ournet);

preds=classify(ournet,augtest);

confusionchart(testds.Labels,preds);

numCorrect=nnz(preds==testds.Labels);

TestAcc=numCorrect/numel(preds);

TestAcc



end

