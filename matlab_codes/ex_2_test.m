%% Pulizia ambiente
clear; clc;close all;

%% Parametri dataset
numSubjects = 2;
numImagesPerSubject = 10;

trainPerSubject = 10;
testPerSubject = numImagesPerSubject - trainPerSubject;

imgHeight = 112;
imgWidth  = 92;

%% Preallocazione tensori e etichette
tensor_train = zeros(imgHeight, imgWidth, trainPerSubject,numSubjects);
tensor_test  = zeros(imgHeight, imgWidth, testPerSubject ,numSubjects);
y_train = zeros(numSubjects * trainPerSubject, 1);
%% Caricamento immagini
for s = 1:numSubjects
    subjectFolder = fullfile(sprintf('p%d',s ));
    
    % Training
    for t = 1:trainPerSubject
        imgPath = fullfile(subjectFolder, sprintf('%d.pgm', t));
        img = double(imread(imgPath));
        tensor_train(:,:,t,s) = img;
        y_train((s-1)*trainPerSubject + t) = s;
    end
    
    % Test
    for t = 1:testPerSubject
        imgPath = fullfile(subjectFolder, sprintf('%d.pgm', t + trainPerSubject));
        img = double(imread(imgPath));
        tensor_test(:,:,t,s) = img;
    end
end

%% Flatten immagini per LDA
X_train = reshape(tensor_train, imgHeight*imgWidth, numSubjects*trainPerSubject)';
X_test  = reshape(tensor_test,  imgHeight*imgWidth, numSubjects*testPerSubject)';

X_train = X_train';
X_test = X_test';

[U,S,V] = svds(X_train,trainPerSubject*numSubjects) ;

Z = U'*X_train;

model = myLDA(Z,y_train);

w = model.W;

Q = U*w;

X1 = X_train(:,1:trainPerSubject);
X_1_mean = mean(X1,2);

subplot(1,3,1)
imshow(reshape(X_1_mean,112,92),[])


subplot(1,3,2)
o = reshape(Q,112,92);
imshow(o,[])

subplot(1,3,3)
imshow(reshape(mean(X_train(:,trainPerSubject+1:end),2),112,92),[])


P = Q'*X_train;
P = P /norm(P);
figure 
histogram(P(1:trainPerSubject)');
hold on 
histogram(P(trainPerSubject+1:end)')
