% script to test the face recognition implementation.
% In this test case we test the meaning of the threshold.
% result: the threshold has influence over the result.

close all; % close all windows
clear all; % clear all variables
clc;       % clear the console

M_1 = 40;
M_2 = 1;

run('faceRecognition');

for jj=0.1:0.05:0.5
    threshold = jj;
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
    [threshold, known, unknown]
end

