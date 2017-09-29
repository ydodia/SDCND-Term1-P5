# Vehicle Tracking - P5

### Overview
The aim of this project was to return to computer vision once more to recognize and track vehicles in a video feed. My pipeline consists of breaking each image into 3 color channels and extracting feature vectors using a sliding window technique. The feature vectors use Histogram Of Gradients (HOG) properties to help classify and predict potential vehicles. Once a vehicle is detected, it's bounding box is calculated and drawn onto each frame. Additionally, I've incorporated Project 4's lane finding algorithm into a combined pipeline.

#### 1 | Training Using SVM
The support vector machine (SVM) was trained using the supplied vehicle and non-vehicle images and with the `sklearn` library and specifically the general `SVC` function with the following parameters:`C=10`, `gamma=0.01` and `kernel='poly'`. The cost value was chosen experimentally as it resulted in a higher accuracy on the test portion of data.

I was able to get a reasonable tracking pipeline using the `YCrCb` color space, and the following HOG parameters: `12 orientations`, `16 pixels per cell`, `2 cells per block`.

#### 2 | Color Channel Selection
I tested out my `train_SVC` function using YCrCb, HLS, HSV and RGB color channels, narrowing down to YCrCb and HLS. Ultimately, YCrCb showed better performance in training/testing, test accuracy of `0.9952`, as well as better detection and stability in the video.

#### 3 | HOG 
The first step in processing an image frame would be to get the HOG features. We don't need to process the entire image dimensions, and instead bounded the image to roughly half of the `y` and cut off about a third of the `x` dimensions. The image below shows an example of obtained HOG features relative to the original image.

[HOG_example]:./HOG_example.png
![Original and HOG images][HOG_example]

#### 4 | Sliding Window Search
We use a sliding window technique to extract various boxes, with a set range of scales, to obtain the features used to predict whether a vehicle is present. For example, the image below shows the sliding boxes for a scale of `1.0`, where blue denotes boxes as they search and predict and a red box denotes a found vehicle.

[sliding_window_1]:./sliding_window_1.png
![Sliding window with single scale][sliding_window_1]

This sliding window technique is used for a variety of scales. In particular, I used scale values in the set `[1.25, 1.5]` with a `1 cell per step` for the HOG tranform. This would result in a full search as given below:

[sliding_window_2]:./sliding_window_2.png
![Sliding window with multiple scales][sliding_window_2]

#### 5 | Heatmap
Once the detected vehicle boxes are found, we need to produce a heatmap and threshold out any erroneous artifacts (false positives). This is done simply by adding up the bounding boxes found and setting a manual threshold to drop the regions where a low number of boxes were found. An example of a resulting heatmap is given below.

[heatmap]:./heatmap.png
![Heatmap example][heatmap]

#### 6 | Obtaining Car Labels
The label function from `scipy` allows us to easily grab and 'label' each car that was found. This function effectively labels each distince block as a bounding box for a found car, after heatmap thresholding. The finished image, with boxes drawn around the found cars looks like the following image:

[found_car]:./found_car.png
![Found cars][found_car]

#### 7 | Finished Pipeline
The pipeline works well against the simple project video. An example output is given below, showing the highlighted lane:

[pipeline_example]:./pipeline_output.png
![Example Output from Pipeline][pipeline_example]

I incorporated the P4 Lane Finding algorithm into this project as well. A 15s snippet of my processed `project_video` is shown below (GIF):

[final]:./project_video_output.gif
![Example Output from Pipeline][final]

The [Full MP4 video](./project_video_output.mp4) is linked from my GitHub.

### Conclusion
I ran into several difficulties; in particular, with image pre-processing, choosing the `x,y` range for the sliding window search for a given scale, choosing the best color space, and reducing noisy artifacts and false positives.

Image pre-processing is critical and I'd like to have spent a bit more time on that.

The HOG parameters were critical to obtaining quality predictions and presented a tradeoff between computational efficiency and prediction accuracy, as expected.

In the future, I would focus on two areas: better image pre-processing, better SVM fit and HOG feature extraction. These work in concert and a good color space will feed into a better SVM which will undoubtedly make for better HOG paramter tuning.





