run('faceRecognition');

% show last 9 eigenface in the training set
figure;
number_of_eigenfaces_to_show = 9;
index = M - number_of_eigenfaces_to_show;
for ii=1:number_of_eigenfaces_to_show
    eigenface = reshape(eigenfaces(:,index+ii),P,Q);
    subplot(3,3,ii);
    imshow(uint8(eigenface));
    title(num2str(ii));
end