# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_non_car.png
[image2]: ./output_images/HOG.png
[image3]: ./output_images/find_cars.png
[image4]: ./output_images/window_scale_1.png
[image5]: ./output_images/window_scale_2.png
[image6]: ./output_images/window_scale_3.png
[image7]: ./output_images/window_scale_4.png
[image8]: ./output_images/find_cars_multiple_sized.png
[image9]: ./output_images/heatmap.png
[image10]: ./output_images/heatmap_thresholded.png
[image11]: ./output_images/labelled.png
[image12]: ./output_images/labelled_cars.png
[image13]: ./output_images/test_images_output.png
[video1]: ./project_video_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The p5_pipeline.py file is where images are worked upon to create the concept for the final video.

First, the car and non-car images are loaded from the provided dataset. A random sample of the images is shown below:

![][image1]

We have a similar number of car and non-car images. This will make sure that the trained SVC is not biased towards one or the other label.

The utils.py file generally contains helper functions used in the main files.

The code for extracting the HOG features is present in the utils.py file.

The figure below shows the Histogram of Oriented Gradients (HOG) of an example Car and Non-Car image.

![][image2]

The extract_features function (also under utils.py), accepts a list of image paths, HOG parameters and one of a number of color spaces to which the image input is converted. It returns a flattened array of HOG features for each of the images.

####2. Explain how you settled on your final choice of HOG parameters.

Various combinations of parameters were tried. The accuracy of the of the classifier making the predictions along with the speed at which it makes the predictions were analyzed to decide upon the best combination.

The best configuration used was: YUV color space, 11 orientations, 16 'pixels per cell', 2 'cells per block', and ALL channels of the 'color space'. 

Using all channels of the 'color space' increases the execution time but also brings an increase in the accuracy. The lost execution time was partly gained by increasing the 'pixels per cell' parameter from 8 to 16.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The training data was split into to train the Classifier (80%) and the rest to test the classifier.

I trained a linear SVM using the default attributes using only the HOG features (without using color histogram and spatial binning) with an accuracy of 98.2%.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The 'find_cars' function (in utils.py) combines HOG feature extraction along with a sliding window search algorithm. The HOG features are extracted for the entire image (or a selected part) and then these features are sub-sampled according to the size of the window and then fed to the classifier. 
The classifier then makes a prediction and a list of rectangles corresponding to a 'car' prediction is returned in the function.

The image below shows the results using a single scale of window size on an example image:

![alt text][image3]

In the function 'find_cars_multiple_sized_windows', several scales for the window sizes were explored. A final configuration of small(1x), medium (1.5x, 2x) and large(3x) windows were seen to best fit the needs.

![][image4]

![][image5]

![][image6]

![][image7]

Smaller scales for window size (0.5x) were also explored. These, however, led to an increase in false positives. An overlap of 75% in the 'y' direction and 50% in the 'x' direction gave the bast returns.
To be able to limit the number of windows to be searched through, a certain range was defined for every scale.  190 window locations were considered in the final implementation, which proved to be robust enough to reliably detect vehicles while maintaining a high speed of execution.

The image below shows the final result on an example image. Notice that there are several positive predictions on the cars in front. It is possible to get some false positives at this stage as shown in the following example image.

![][image8]

Since a correct prediction of a vehicle would involve multiple detections in the space of the vehicle, the false positives can be minimized using an 'add_heat' function (under utils.py). This function increments the pixel value (heat) of all the pixels detected within a rectangle by 1 in an all-black image of the size of the original image. Thus, multiplle detections in the same space results in more heat for the pixels.

The following image shows this functionality corresponding to the image above.

![][image9]

A threshold is applied (function 'apply_threshold') to the heatmap setting all pixels not exceeding a certain threshold (1 in the example below) to zero.

![][image10]

The scipy.ndimage.measurements.label() function collects spatially contiguous areas of the heatmap and assigns each a label:

![][image11]

The final detection area is the maximum of the labelled boxes.

![][image12]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The test images give the following results:

![][image13]

Optimization techniques included changes to window sizing and overlap as already described, and lowering the heatmap threshold to improve accuracy of the detection (higher threshold values tended to underestimate the size of the vehicle).
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The video was processed in the 'p5_video_advanced.py' file. A '_process_frame_for_video' function was used to combine all the steps defined above. 

All the detected rectangles from the previous 15 frames of the video were stored in a separate class ('RectanglesList'). The threshold for the heatmap was then set to '1 + len(previous_rectangles.rects)//2'. This number was decided upon through trial and error. The heatmap always looked better when using a threshold based on the length of the detected rectangles from the previous frames.  

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1. Calculating hog features is an expensive operation as the gradients of every pixel must be calculated. However, by calculating hog features once for the entire image and then subsampling that with our sliding windows, we prevent any duplicated work.

2. cv2 reads in images with pixel values 0 < val < 255, whereas mpimg reads in png at 0 < val < 1 and jpg at 0 < val < 255. Furthermore, cv2 reads in colors as BGR, whereas mpimg reads them in as RGB. I used mpimg for png and normalized with 255 when using cv2.

3. Further reading needs to be done to get a better intuition of the different color spaces that can be used.

4. Bounding box needs to be improved. Perhaps by tracking the position and velocity, the position could be predicted for the next frame using a Kalman Filter, for example.

5. This algorithm is highly sensitive to light conditions and may fail under different light conditions.

6. Deep learning could provide better results for vehicle recognition, perhaps using ['You only look once'](https://pjreddie.com/darknet/yolo/) or ['SqueezeDet'](https://arxiv.org/abs/1612.01051)

7. The pipeline used in this project tends to do poorly when areas of the image darken by the presence of shadows. Areas of dark pixels are often classified as cars, creating false-positives. This issue could be resolved by adding more dark images to the non-vehicle dataset.

8. 'xstart' and 'xstop' could also be incorporated into the code to leave out the unused parts of the frame. This will improve the performance and reduce false positives.

9. At present, the maximum of the labelled boxes are being used to creaate the bounding box. This could be optimized by setting expected proportions for a car. 

### References

1. https://github.com/ILYAmLV/Vehicle-Detection-and-Tracking
2. https://github.com/sumitbinnani/CarND-Vehicle-Detection
3. https://github.com/BillZito/vehicle-detection
4. https://github.com/xmprise/Vehicle_Detection_and_Tracking

