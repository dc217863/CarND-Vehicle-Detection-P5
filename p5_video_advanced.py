from sklearn.model_selection import train_test_split
from moviepy.editor import VideoFileClip
from sklearn.svm import LinearSVC
from scipy.ndimage.measurements import label
from sklearn.preprocessing import StandardScaler
import glob
import time

from utils import *


# Load Training Data
car_images = glob.glob('training_sets/vehicles/vehicles/**/*.png')
non_car_images = glob.glob('training_sets/non-vehicles/non-vehicles/**/*.png')

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
      'pixels per cell and', cell_per_block, 'cells per block')
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


class RectanglesList(object):
    """ class to store a list of rectangles from video """
    def __init__(self):
        self.rects = list()

    def add_rects(self, rects):
        self.rects.append(rects)
        while len(self.rects) > 15:
            self.rects.pop(0)

    def clean_rects(self):
        self.rects[:] = []


# history of rectangles in previous frames
previous_rectangles = RectanglesList()


def _process_frame_for_video(img):
    rectangles = find_cars_multiple_sized_windows(img, svc, colorspace, X_scaler, orient, pix_per_cell,
                                                  cell_per_block, spatial_size, hist_bins)
    if len(rectangles) > 0:
        previous_rectangles.add_rects(rectangles)

    heatmap_img = np.zeros_like(img[:, :, 0])
    for rect_set in previous_rectangles.rects:
        heatmap_img = add_heat(heatmap_img, rect_set)
    heatmap_img = apply_threshold(heatmap_img, 1 + len(previous_rectangles.rects) // 2)
    labels = label(heatmap_img)
    draw_img, rects = draw_labeled_bboxes(np.copy(img), labels)
    return draw_img


print("Video1...")
test_out_file = 'test_video_out.mp4'
clip_test = VideoFileClip('test_video.mp4')
clip_test_out = clip_test.fl_image(_process_frame_for_video)
clip_test_out.write_videofile(test_out_file, audio=False)

previous_rectangles.clean_rects()

print("Video2...")
proj_out_file = 'project_video_out_h2.mp4'
clip_proj = VideoFileClip('project_video.mp4')
clip_proj_out = clip_proj.fl_image(_process_frame_for_video)
clip_proj_out.write_videofile(proj_out_file, audio=False)
