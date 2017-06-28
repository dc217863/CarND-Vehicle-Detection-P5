import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from scipy.ndimage.measurements import label
import glob
import time

from utils import *

# Load Training Data
car_images = glob.glob('training_sets/vehicles/vehicles/**/*.png')
non_car_images = glob.glob('training_sets/non-vehicles/non-vehicles/**/*.png')
visualize_training_data(car_images, non_car_images)


# using test image
car_img = mpimg.imread(car_images[50])
_, car_dst = get_hog_features(car_img[:, :, 2], 9, 8, 8, vis=True, feature_vec=True)
non_car_img = mpimg.imread(non_car_images[50])
_, non_car_dst = get_hog_features(non_car_img[:, :, 2], 9, 8, 8, vis=True, feature_vec=True)
visualize_hog(car_img, car_dst, non_car_img, non_car_dst)


# Feature extraction parameters
colorspace = 'YUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11
pix_per_cell = 16
cell_per_block = 2
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)
hist_bins = 32

print('Extracting features from images...')
t1 = time.time()
car_features = extract_features(car_images, color_space=colorspace, orient=orient,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=False,
                                hist_feat=False, hog_feat=True)
non_car_features = extract_features(non_car_images, color_space=colorspace, orient=orient,
                                   spatial_size=spatial_size, hist_bins=hist_bins,
                                   pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block,
                                   hog_channel=hog_channel, spatial_feat=False,
                                   hist_feat=False, hog_feat=True)
t2 = time.time()
print(round(t2-t1, 2), 'Seconds to extract HOG features...')
# Create an array stack of feature vectors
X = np.vstack((car_features, non_car_features)).astype(np.float64)

# Fit a per-column scaler - necessary if combining different types of features (HOG + color_hist + bin_spatial)
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=rand_state)

print('Using:', orient, 'orientations', pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))


# Use a linear SVC
svc = LinearSVC()
print('Training Classifier...')
# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t1 = time.time()
n_predict = 10
print('My SVC predicts:     ', svc.predict(X_test[0:n_predict]))
print('For these', n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t1, 5), 'Seconds to predict', n_predict, 'labels with SVC')


test_img = mpimg.imread('test_images/test4.jpg')
ystart = 400
ystop = 656
scale = 1.5
# colorspace = 'YUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb


print("Finding cars in an image...")
rectangles = find_cars(test_img, ystart, ystop, scale, colorspace, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                    hist_bins)
out_img = draw_boxes(test_img, rectangles)
plt.figure(figsize=(10, 10))
plt.title('Finding cars in an image')
plt.imshow(out_img)


print("Finding cars in an image with multiple sized windows ...")
rectangles = find_cars_multiple_sized_windows(test_img, svc, colorspace, X_scaler, orient, pix_per_cell, cell_per_block,
                                              spatial_size, hist_bins)
test_img_rects = draw_boxes(test_img, rectangles, color='random', thick=2)
plt.figure(figsize=(10, 10))
plt.title('Finding cars in an image with multiple sized windows')
plt.imshow(test_img_rects)
print('Number of boxes: ', len(rectangles))


# Test out the heatmap
heatmap_img = np.zeros_like(test_img[:, :, 0])
heatmap_img = add_heat(heatmap_img, rectangles)
plt.figure(figsize=(10, 10))
plt.title('Heatmap')
plt.imshow(heatmap_img, cmap='hot')


thresholded_heatmap_img = apply_threshold(heatmap_img, 1)
plt.figure(figsize=(10, 10))
plt.title('Thresholded Heatmap')
plt.imshow(thresholded_heatmap_img, cmap='hot')


labels = label(heatmap_img)
plt.figure(figsize=(10, 10))
plt.title('Labelled image')
plt.imshow(labels[0], cmap='gray')
print(labels[1], 'cars found')


# Draw bounding boxes on a copy of the image
draw_img, rect = draw_labeled_bboxes(np.copy(test_img), labels)
plt.figure(figsize=(10, 10))
plt.imshow(draw_img)

plt.show()
