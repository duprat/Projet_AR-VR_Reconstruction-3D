imgL = imread('L.jpg');
imgR = imread('R.jpg');

%imageDir = fullfile(toolboxdir('vision'), 'visiondata','upToScaleReconstructionImages');
%images = imageDatastore(imageDir);
%imgL = readimage(images, 1);
%imgR = readimage(images, 2);

%figure
%imshowpair(imgL, imgR, 'montage'); title('Original Images');

%load upToScaleReconstructionCameraParameters.mat;

%cameraCalibrator;

load calibrationSession.mat;

imgL = undistortImage(imgL, cameraParams);
imgR = undistortImage(imgR, cameraParams);

%figure 
%imshowpair(imgL, imgR, 'montage');
%title('Undistorted Images');

imagePoints1 = detectMinEigenFeatures(rgb2gray(imgL), 'MinQuality', 0.1);

figure
imshow(imgL, 'InitialMagnification', 50);
title('150 Strongest Corners from the First Image');
hold on
plot(selectStrongest(imagePoints1, 150));

tracker = vision.PointTracker('MaxBidirectionalError', 1, 'NumPyramidLevels', 5);

imagePoints1 = imagePoints1.Location;
initialize(tracker, imagePoints1, imgL);

[imagePoints2, validIdx] = step(tracker, imgR);
matchedPoints1 = imagePoints1(validIdx, :);
matchedPoints2 = imagePoints2(validIdx, :);

figure
showMatchedFeatures(imgL, imgR, matchedPoints1, matchedPoints2);
title('Tracked Features');

[fMatrix, epipolarInliers] = estimateFundamentalMatrix(...
  matchedPoints1, matchedPoints2, 'Method', 'MSAC', 'NumTrials', 10000);

inlierPoints1 = matchedPoints1(epipolarInliers, :);
inlierPoints2 = matchedPoints2(epipolarInliers, :);

figure
showMatchedFeatures(imgR, imgL, inlierPoints1, inlierPoints2);
title('Epipolar Inliers');

[R, t] = cameraPose(fMatrix, cameraParams, inlierPoints1, inlierPoints2);

imagePoints1 = detectMinEigenFeatures(rgb2gray(imgL), 'MinQuality', 0.001);

tracker = vision.PointTracker('MaxBidirectionalError', 1, 'NumPyramidLevels', 5);

imagePoints1 = imagePoints1.Location;
initialize(tracker, imagePoints1, imgL);

[imagePoints2, validIdx] = step(tracker, imgR);
matchedPoints1 = imagePoints1(validIdx, :);
matchedPoints2 = imagePoints2(validIdx, :);

camMatrix1 = cameraMatrix(cameraParams, eye(3), [0 0 0]);
camMatrix2 = cameraMatrix(cameraParams, R', -t*R');

points3D = triangulate(matchedPoints1, matchedPoints2, camMatrix1, camMatrix2);

numPixels = size(imgL, 1) * size(imgL, 2);
allColors = reshape(imgL, [numPixels, 3]);
colorIdx = sub2ind([size(imgL, 1), size(imgL, 2)], round(matchedPoints1(:,2)), ...
    round(matchedPoints1(:, 1)));
color = allColors(colorIdx, :);

ptCloud = pointCloud(points3D, 'Color', color);

cameraSize = 0.3;
figure
plotCamera('Size', cameraSize, 'Color', 'r', 'Label', '1', 'Opacity', 0);
hold on
grid on
plotCamera('Location', t, 'Orientation', R, 'Size', cameraSize, ...
    'Color', 'b', 'Label', '2', 'Opacity', 0);

pcshow(ptCloud, 'VerticalAxis', 'y', 'VerticalAxisDir', 'down', ...
    'MarkerSize', 45);

camorbit(0, -30);
camzoom(1.5);

xlabel('x-axis');
ylabel('y-axis');
zlabel('z-axis')

title('Up to Scale Reconstruction of the Scene');