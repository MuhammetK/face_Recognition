run('faceRecognition');

figure;
afi = reshape(Psi,P,Q);
imshow(uint8(afi));
title('Average face image');