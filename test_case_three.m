% script to test the face recognition implementation.
% In this test case ...

close all; % close all windows
clear all; % clear all variables
clc;       % clear the console

M_1 = 35;
M_2 = 9;

run('faceRecognition');

threshold = 0.15;

    
known = 0;
unknown = 0;

for ii=(M_1+1):40

    % input image, last image of a person
    test_image = FaceData(ii,10).Image;
    
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
    a=max(similarity_score) * threshold;
    b=min(similarity_score);
    if(b>a)
        unknown = unknown + 1;
    else
        known = known + 1;
        % display the result
        subplot(3,2,4);
        imshow(uint8(reshape(A(:,match_index), P,Q)));title('recognized face');
    end
    [ii, threshold, known, unknown]
end


