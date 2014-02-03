% Face recognition using eigenfaces
%
% This script takes M face images from FaceData_56_46 as the training set.
% Then in the first part the eigenfaces of the training set is calculated.
% In the second part a given face (test_image) will be verified with the
% help of the training set. If the person is an authorized person, the
% script will show the recognized face and for an unauthorized person it
% will show an error message.

close all; % close all windows
clear all; % clear all variables
clc;       % clear the console

%FaceData: load all faces from file. the file contains 40 different persons
% where each person has 10 different face images.
load('FaceData_56_46.mat');

M_1 = 9; % number of different persons
M_2 = 6; % number of test images per person
M = M_1 * M_2;   % total number of test images


% take an test image to get the height and weight
TMP_Image = FaceData(M_1,M_2).Image;
[P,Q] = size(TMP_Image);
R = P*Q;

A=zeros(R,M); % training matrix; 
A_column_index = 1;
for ii=1:M_1
    for jj=1:M_2
        Im = (FaceData(ii,jj).Image);
        
        % transform matrix into an vector
        Gamma_i = reshape(Im,R,1);
        
        A(:,A_column_index) = double(Gamma_i);
        A_column_index = A_column_index + 1;
    end
end


% computing the average face vector Psi
sum = 0;
for ii=1:M
    sum = sum + A(:,ii);
end
Psi = sum/M;


% subtracing the mean face from each face images in the training set: 
% Phi_i = Gamma_i - Psi 
Phi = zeros(R,M);
for ii=1:M
    Phi(:,ii) = A(:,ii) - Psi;
end


% computing the covariance matrix C:;
C = Phi' * Phi; 

[eigenvectors, eigenvalues] = eig(C);


% computing eigenfaces, the eigenvectors of Phi*Phi'
eigenfaces = Phi * eigenvectors;



weights = zeros(M,M);
for jj=1 : M
    for ii=1 : M
        weights(ii,jj)= eigenfaces(:,ii)' * (A(:,jj)-Psi);
    end
end


% --------------------------------------------------
% --------------------------------------------------
% ----------------- Face Recognition ---------------
% --------------------------------------------------
% --------------------------------------------------

% input image
test_image = FaceData(5,10).Image;

figure,
subplot(3,2,3);
imshow(test_image);title('test face');

% transform the test_image matrix into a vector
v_test_image = double(reshape(test_image,R,1));

test_weights = eigenfaces' * (v_test_image - Psi);

% use inverse Euclidean distance 
similarity_score = arrayfun(@(n) 1 / (1 + norm(weights(:,n) - test_weights)), 1:M);

% find the image with the highest similarity
[match_score, match_index] = max(similarity_score);


% decision parameter
a=max(similarity_score) * 0.25;
b=min(similarity_score);
if(b>a)
    msgbox('UNKNOWN FACE','error','error');
    subplot(3,2,4);
    imshow(zeros(P,Q));title('unknown face'); 
else
    % display the result
    subplot(3,2,4);
    imshow(uint8(reshape(A(:,match_index), P,Q)));title('recognized face');
end