
path={'E:\DATASETS\SMIDS\Set1_8xAugmented\fold'};


for i=1:5
 
clear ournet;


char=int2str(i); %for döngüsünden 1 yerine i olcak.    
trainconcat=strcat(char,'\train');
trainpath=strcat(path,trainconcat);


imds = imageDatastore(trainpath,'IncludeSubfolders',true,'LabelSource','foldernames');


augimds=augmentedImageDatastore([150 150 3],imds);


testconcat=strcat(char,'\test');
testpath=strcat(path,testconcat);




testds=imageDatastore(testpath,'IncludeSubfolders',true,'LabelSource','foldernames');
augtest=augmentedImageDatastore([150 150 3],testds);

lgraph=googlenetLayers();


valFreq=floor(numel(augimds.Files)/16);

options=trainingOptions('adam','MaxEpochs',75,'InitialLearnRate',0.001,'ValidationData',augtest,'ValidationFrequency',valFreq,...
    'Verbose',true,'Shuffle','every-epoch','Plots','training-progress','MiniBatchSize',16,'ExecutionEnvironment','gpu',...
    'LearnRateSchedule','piecewise','LearnRateDropFactor',0.1,'LearnRateDropPeriod',999);

tic
ournet=trainNetwork(augimds,lgraph,options);  
toc

analyzeNetwork(ournet);

preds=classify(ournet,augtest);

confusionchart(testds.Labels,preds);

numCorrect=nnz(preds==testds.Labels);

TestAcc=numCorrect/numel(preds);

TestAcc



end












function lgraph = googlenetLayers()


lgraph = layerGraph();


tempLayers = [
    imageInputLayer([150 150 3],"Name","data")
    convolution2dLayer([3 3],32,"Name","a_1","Padding",[1 1 1 1],"BiasLearnRateFactor",2)
    reluLayer("Name","a_2")   
    maxPooling2dLayer([3 3],"Name","a_5","Padding",[1 1 1 1],"Stride",3)
    crossChannelNormalizationLayer(5,"Name","a_6","K",1)
    convolution2dLayer([3 3],32,"Name","a_7_2","Padding",[1 1 1 1],"BiasLearnRateFactor",2)
    reluLayer("Name","a_8_2")
    crossChannelNormalizationLayer(5,"Name","a_11","K",1)
    maxPooling2dLayer([3 3],"Name","pool2-3x3_s2","Padding",[1 1 1 1],"Stride",3)];
lgraph = addLayers(lgraph,tempLayers); 

%BLOCK 0

tempLayers = [
    convolution2dLayer([1 1],8,"Name","inception_3a-5x5_reduce","BiasLearnRateFactor",2)
    reluLayer("Name","inception_3a-relu_5x5_reduce")
    convolution2dLayer([5 5],16,"Name","inception_3a-5x5","Padding",[2 2 2 2],"BiasLearnRateFactor",2)
    reluLayer("Name","inception_3a-relu_5x5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","inception_3a-1x1","BiasLearnRateFactor",2)
    reluLayer("Name","inception_3a-relu_1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","inception_3a-3x3_reduce","BiasLearnRateFactor",2)
    reluLayer("Name","inception_3a-relu_3x3_reduce")
    convolution2dLayer([3 3],32,"Name","inception_3a-3x3","Padding",[1 1 1 1],"BiasLearnRateFactor",2)
    reluLayer("Name","inception_3a-relu_3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Name","inception_3a-pool","Padding",[1 1 1 1])
    convolution2dLayer([1 1],32,"Name","inception_3a-pool_proj","BiasLearnRateFactor",2)
    reluLayer("Name","inception_3a-relu_pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(4,"Name","inception_3a-output");
lgraph = addLayers(lgraph,tempLayers);
%BLOCK1



tempLayers = [
   convolution2dLayer([1 1],32,"Name","inception_3b-3x3_reduce","BiasLearnRateFactor",2)
   reluLayer("Name","inception_3b-relu_3x3_reduce")
   convolution2dLayer([3 3],32,"Name","inception_3b-3x3","Padding",[1 1 1 1],"BiasLearnRateFactor",2)
   reluLayer("Name","inception_3b-relu_3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","inception_3b-1x1","BiasLearnRateFactor",2)
    reluLayer("Name","inception_3b-relu_1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Name","inception_3b-pool","Padding",[1 1 1 1])
    convolution2dLayer([1 1],32,"Name","inception_3b-pool_proj","BiasLearnRateFactor",2)
    reluLayer("Name","inception_3b-relu_pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],8,"Name","inception_3b-5x5_reduce","BiasLearnRateFactor",2)
    reluLayer("Name","inception_3b-relu_5x5_reduce")
    convolution2dLayer([5 5],16,"Name","inception_3b-5x5","Padding",[2 2 2 2],"BiasLearnRateFactor",2)
    reluLayer("Name","inception_3b-relu_5x5")];
lgraph = addLayers(lgraph,tempLayers);

%BLOCK2 without depthconcatenation

tempLayers = [
    depthConcatenationLayer(4,"Name","inception_3b-output")
    maxPooling2dLayer([3 3],"Name","pool3-3x3_s2","Padding",[1 1 1 1],"Stride",3)];
lgraph = addLayers(lgraph,tempLayers);





tempLayers = [
    convolution2dLayer([1 1],32,"Name","inception_4a-3x3_reduce","BiasLearnRateFactor",2)
    reluLayer("Name","inception_4a-relu_3x3_reduce")
    convolution2dLayer([3 3],32,"Name","inception_4a-3x3","Padding",[1 1 1 1],"BiasLearnRateFactor",2)
    reluLayer("Name","inception_4a-relu_3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","inception_4a-1x1","BiasLearnRateFactor",2)
    reluLayer("Name","inception_4a-relu_1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],8,"Name","inception_4a-5x5_reduce","BiasLearnRateFactor",2)
    reluLayer("Name","inception_4a-relu_5x5_reduce")
    convolution2dLayer([5 5],16,"Name","inception_4a-5x5","Padding",[2 2 2 2],"BiasLearnRateFactor",2)
    reluLayer("Name","inception_4a-relu_5x5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Name","inception_4a-pool","Padding",[1 1 1 1])
    convolution2dLayer([1 1],32,"Name","inception_4a-pool_proj","BiasLearnRateFactor",2);
    reluLayer("Name","inception_4a-relu_pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(4,"Name","inception_4a-output");
lgraph = addLayers(lgraph,tempLayers);

%Block 3



tempLayers = [
    maxPooling2dLayer([3 3],"Name","inception_4b-pool","Padding",[1 1 1 1])
    convolution2dLayer([1 1],32,"Name","inception_4b-pool_proj","BiasLearnRateFactor",2)
    reluLayer("Name","inception_4b-relu_pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
   convolution2dLayer([1 1],64,"Name","inception_4b-1x1","BiasLearnRateFactor",2)
   reluLayer("Name","inception_4b-relu_1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","inception_4b-3x3_reduce","BiasLearnRateFactor",2)
    reluLayer("Name","inception_4b-relu_3x3_reduce")
    convolution2dLayer([3 3],64,"Name","inception_4b-3x3","Padding",[1 1 1 1],"BiasLearnRateFactor",2)
    reluLayer("Name","inception_4b-relu_3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],16,"Name","inception_4b-5x5_reduce","BiasLearnRateFactor",2)
    reluLayer("Name","inception_4b-relu_5x5_reduce")
    convolution2dLayer([5 5],32,"Name","inception_4b-5x5","Padding",[2 2 2 2],"BiasLearnRateFactor",2)
    reluLayer("Name","inception_4b-relu_5x5")];
lgraph = addLayers(lgraph,tempLayers);


tempLayers = depthConcatenationLayer(4,"Name","inception_4b-output");
lgraph = addLayers(lgraph,tempLayers);





%Block 4 end



tempLayers = [
    convolution2dLayer([1 1],16,"Name","inception_4c-5x5_reduce","BiasLearnRateFactor",2)
    reluLayer("Name","inception_4c-relu_5x5_reduce")
    convolution2dLayer([5 5],32,"Name","inception_4c-5x5","Padding",[2 2 2 2],"BiasLearnRateFactor",2)
    reluLayer("Name","inception_4c-relu_5x5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","inception_4c-1x1","BiasLearnRateFactor",2)
    reluLayer("Name","inception_4c-relu_1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Name","inception_4c-pool","Padding",[1 1 1 1])
    convolution2dLayer([1 1],32,"Name","inception_4c-pool_proj","BiasLearnRateFactor",2)
    reluLayer("Name","inception_4c-relu_pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","inception_4c-3x3_reduce","BiasLearnRateFactor",2)
    reluLayer("Name","inception_4c-relu_3x3_reduce")
    convolution2dLayer([3 3],64,"Name","inception_4c-3x3","Padding",[1 1 1 1],"BiasLearnRateFactor",2)
    reluLayer("Name","inception_4c-relu_3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(4,"Name","inception_4c-output");
lgraph = addLayers(lgraph,tempLayers);



tempLayers = [
    convolution2dLayer([3 3],32,"Name","c0","BiasLearnRateFactor",2,"Padding",[1 1 1 1])
    maxPooling2dLayer([3 3],"Name","mp","Padding",[1 1 1 1],"Stride",3)
    fullyConnectedLayer(512,"Name","fc1","BiasLearnRateFactor",2)
    reluLayer("Name","r0")
    fullyConnectedLayer(3,"Name","loss3-classifier","BiasLearnRateFactor",2)
    softmaxLayer("Name","prob")
    classificationLayer("Name","output")];
lgraph = addLayers(lgraph,tempLayers);


%--------------------------------------------------------------------------

lgraph = connectLayers(lgraph,"pool2-3x3_s2","inception_3a-5x5_reduce");
lgraph = connectLayers(lgraph,"pool2-3x3_s2","inception_3a-1x1");
lgraph = connectLayers(lgraph,"pool2-3x3_s2","inception_3a-3x3_reduce");
lgraph = connectLayers(lgraph,"pool2-3x3_s2","inception_3a-pool");
lgraph = connectLayers(lgraph,"inception_3a-relu_1x1","inception_3a-output/in1");
lgraph = connectLayers(lgraph,"inception_3a-relu_5x5","inception_3a-output/in3");
lgraph = connectLayers(lgraph,"inception_3a-relu_pool_proj","inception_3a-output/in4");
lgraph = connectLayers(lgraph,"inception_3a-relu_3x3","inception_3a-output/in2");
%Block1 end




lgraph = connectLayers(lgraph,"inception_3a-output","inception_3b-3x3_reduce");
lgraph = connectLayers(lgraph,"inception_3a-output","inception_3b-1x1");
lgraph = connectLayers(lgraph,"inception_3a-output","inception_3b-pool");
lgraph = connectLayers(lgraph,"inception_3a-output","inception_3b-5x5_reduce");
lgraph = connectLayers(lgraph,"inception_3b-relu_pool_proj","inception_3b-output/in4");
lgraph = connectLayers(lgraph,"inception_3b-relu_1x1","inception_3b-output/in1");
lgraph = connectLayers(lgraph,"inception_3b-relu_5x5","inception_3b-output/in3");
lgraph = connectLayers(lgraph,"inception_3b-relu_3x3","inception_3b-output/in2");
%Block2 end 





lgraph = connectLayers(lgraph,"pool3-3x3_s2","inception_4a-3x3_reduce");
lgraph = connectLayers(lgraph,"pool3-3x3_s2","inception_4a-1x1");
lgraph = connectLayers(lgraph,"pool3-3x3_s2","inception_4a-5x5_reduce");
lgraph = connectLayers(lgraph,"pool3-3x3_s2","inception_4a-pool");
lgraph = connectLayers(lgraph,"inception_4a-relu_1x1","inception_4a-output/in1");
lgraph = connectLayers(lgraph,"inception_4a-relu_3x3","inception_4a-output/in2");
lgraph = connectLayers(lgraph,"inception_4a-relu_pool_proj","inception_4a-output/in4");
lgraph = connectLayers(lgraph,"inception_4a-relu_5x5","inception_4a-output/in3");
%Block3 end



lgraph = connectLayers(lgraph,"inception_4a-output","inception_4b-pool");
lgraph = connectLayers(lgraph,"inception_4a-output","inception_4b-1x1");
lgraph = connectLayers(lgraph,"inception_4a-output","inception_4b-3x3_reduce");
lgraph = connectLayers(lgraph,"inception_4a-output","inception_4b-5x5_reduce");
lgraph = connectLayers(lgraph,"inception_4b-relu_1x1","inception_4b-output/in1");
lgraph = connectLayers(lgraph,"inception_4b-relu_3x3","inception_4b-output/in2");
lgraph = connectLayers(lgraph,"inception_4b-relu_pool_proj","inception_4b-output/in4");
lgraph = connectLayers(lgraph,"inception_4b-relu_5x5","inception_4b-output/in3");
%Block4 end





lgraph = connectLayers(lgraph,"inception_4b-output","inception_4c-5x5_reduce");
lgraph = connectLayers(lgraph,"inception_4b-output","inception_4c-1x1");
lgraph = connectLayers(lgraph,"inception_4b-output","inception_4c-pool");
lgraph = connectLayers(lgraph,"inception_4b-output","inception_4c-3x3_reduce");
lgraph = connectLayers(lgraph,"inception_4c-relu_1x1","inception_4c-output/in1");
lgraph = connectLayers(lgraph,"inception_4c-relu_5x5","inception_4c-output/in3");
lgraph = connectLayers(lgraph,"inception_4c-relu_3x3","inception_4c-output/in2");
lgraph = connectLayers(lgraph,"inception_4c-relu_pool_proj","inception_4c-output/in4");

%Block5 end

lgraph = connectLayers(lgraph,"inception_4c-output","c0");





end