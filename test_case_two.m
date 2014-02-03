% script to test the face recognition implementation.
% In this test case we test the meaning of the training set. therefore I 
% variegate the number of images of one person in the training set.
% result: the more images of one person is in the training set the merrier 
% higher is the recognition rate.

close all; % close all windows
clear all; % clear all variables
clc;       % clear the console

M_1 = 40;

threshold = 0.26;
for p=1:9
    M_2 = p;

    run('faceRecognition');

    known = 0;
    unknown = 0;
    
    for ii=1:40

        % input image, last image of a person
        test_image = FaceData(ii,10).Image;


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
        end
    end
    [p, threshold, known, unknown]
end

