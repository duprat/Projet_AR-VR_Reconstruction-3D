v = VideoReader('C:\Users\Thomas\Data\GithubRepos\Projet_AR-VR_Reconstruction-3D\Thomas\Videos\blue.mp4');

accu = 0;
while hasFrame(v)
    
    img = readFrame(v);
    
    imwrite(img, strcat('C:\Users\Thomas\Data\GithubRepos\Projet_AR-VR_Reconstruction-3D\Thomas\Images\FromVid\',num2str(accu),'.jpg'));
    accu = accu + 1;
end