### Various module imports ###
import numpy as np
import cv2
import glob
import time

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.svm import LinearSVC


# Some constants
COLOR_RED=(255,0,0)
COLOR_GREEN=(0,255,0)
COLOR_BLUE=(0,0,255)
COLOR_ORANGE=(255,127,0)
COLOR_PURPLE=(127,0,255)
PICKLE_FILE='training_data/saved_model.p'



####### Globals ########
# reference to the model and scaler needed after training.
# these are read in from the trained model that is saved as a pickle file
svc=None
X_scaler=None


#One place to define the various parameters
def get_the_parameters():
    color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 8  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = 0  # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16)  # Spatial binning dimensions
    hist_bins = 16  # Number of histogram bins
    spatial_feat = True  # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = True  # HOG features on or off
    return (color_space,
            orient, pix_per_cell, cell_per_block, hog_channel
            ,spatial_size, hist_bins
            , spatial_feat, hist_feat, hog_feat)


#Find all the car and not_car image names from the training_data folder
def car_notcar_image_names():
    car_images = ['training_data/vehicles/GTI_Far/*.png',
                  'training_data/vehicles/GTI_Left/*.png',
                  'training_data/vehicles/GTI_Right/*.png',
                  'training_data/vehicles/GTI_MiddleClose/*.png',
                  'training_data/vehicles/KITTI_extracted/*.png',
                  ]
    not_car_images = ['training_data/non-vehicles/Extras/*.png',
                      'training_data/non-vehicles/GTI/*.png'
                      ]

    cars = []
    notcars = []

    for car_image in car_images:
        for image in glob.glob(car_image):
            cars.append(image)

    for not_car_image in not_car_images:
        for image in glob.glob(not_car_image):
            notcars.append(image)

    return (cars, notcars)


# Get the HOG features for this channel of image
# Note supplied image is 2d for one color channel
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=True, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=False, feature_vector=feature_vec)
        return features


# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 1)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features



# return a features array of concatenated individual features.
# Note the features array is not normalised
def single_img_features(img, color_space='RGB',
                        spatial_feat=True, spatial_size=(32, 32),
                        hist_feat=True, hist_bins=32,
                        hog_feat=True, orient=8, pix_per_cell=8, cell_per_block=2, hog_channel=0):
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    concat = np.concatenate(img_features)
    return concat




# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features_all_imagenames(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        file_features = single_img_features(image, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
        features.append(file_features)
    # Return list of feature vectors
    return features


#Perform the training on the car and not_car images.
#Take a random sample of 0.80% of the data for the training
#From both the car and not_car images
def train_data_set_and_generate_model():
    import pickle

    (cars,notcars) = car_notcar_image_names()
    print("Car images: ", len(cars)," / Non-car images: ",len(notcars))

    # Reduce the sample size because
    # The quiz evaluator times out after 13s of CPU time
    # sample_size = 3000
    # cars = cars[0:sample_size]
    # notcars = notcars[0:sample_size]

    (color_space,
     orient, pix_per_cell, cell_per_block, hog_channel
     , spatial_size, hist_bins
     , spatial_feat, hist_feat, hog_feat) = get_the_parameters()

    car_features = extract_features_all_imagenames (cars, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features_all_imagenames (notcars, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)

    #combine the features for car and not_car images
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t1 = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t1, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t3 = time.time()
    print("Dumping model to pickle file: ", PICKLE_FILE)
    file = open(PICKLE_FILE, "wb")
    pickle.dump((svc,X_scaler), file)
    return (svc, X_scaler)



#Return all the sliding windows given the corner coordinate and the window size.
def get_sliding_windows_for_size (img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list



# Given a set of Sliding windows identify the ones that
# the model predicts as containing a car
# Return the windows that have been identified as containing a car
def search_windows_for_match (img, windows, clf, scaler, color_space='RGB',
                    spatial_size=(32, 32), hist_bins=32,
                    hist_range=(0, 256), orient=9,
                    pix_per_cell=8, cell_per_block=2,
                    hog_channel=0, spatial_feat=True,
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows



# Run the sliding windows for multiple window sizes and start_stop positions
# And return all the windows that identify as having a car in them
# Multiple overlapping windows will be returned by this function.
def multi_sliding_windows(image):
    sliding_window_params = [{'y_start_stop': [360,544], 'xy_overlap':(0.75,0.75), 'xy_window':(64,64), 'color':COLOR_BLUE},
                             {'y_start_stop': [384,544], 'xy_overlap':(0.75, 0.75),'xy_window':(128, 128),'color': COLOR_GREEN},
                             {'y_start_stop': [384,544], 'xy_overlap':(0.8, 0.8),  'xy_window':(160, 160),'color': COLOR_ORANGE},
                             {'y_start_stop': [384,576], 'xy_overlap':(0.75, 0.75),'xy_window':(192, 192),'color': COLOR_PURPLE},
                             {'y_start_stop': [376,640], 'xy_overlap':(0.75, 0.75),'xy_window':(256, 256),'color': COLOR_RED}]

    (color_space,
     orient, pix_per_cell, cell_per_block, hog_channel
     , spatial_size, hist_bins
     , spatial_feat, hist_feat, hog_feat) = get_the_parameters()

    hot_windows_accum=[]
    for param in sliding_window_params:
        x_start_stop=[None,None]
        y_start_stop = param['y_start_stop']
        xy_overlap=param['xy_overlap']
        xy_window=param['xy_window']
        # if (xy_window[0] != 64):
        #     continue
        windows = get_sliding_windows_for_size (image, x_start_stop=x_start_stop, y_start_stop=y_start_stop,
                           xy_window=xy_window, xy_overlap=xy_overlap)

        hot_windows = search_windows_for_match (image, windows, svc, X_scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)
        hot_windows_accum.extend(hot_windows)

    return hot_windows_accum


def debug_show_multiwindows (image, hot_windows, color=(0,0,255), line_thickness=6):
    draw_image = np.copy(image)
    window_img = draw_boxes(draw_image, hot_windows, color=color, line_thickness=4)
    plt.imshow(window_img)
    plt.show()

#######################################################

#Utility function to draw colored outline boxes on the image
def draw_boxes(img, bboxes, color=(0, 0, 255), line_thickness=6):
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(img, bbox[0], bbox[1], color, line_thickness)
    # Return the image copy with boxes drawn
    return img


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap < threshold] = 0
    # Return thresholded map
    return heatmap



def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img




def draw_heat_map(image, box_list):
    from scipy.ndimage.measurements import label
    # Read in image similar to one shown above
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat, box_list)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 1)

    # Visualize the heatmap when displaying
    #heatmap = np.clip(heat, 0, 255)

    heatmap=heat

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    labelled_img = draw_labeled_bboxes(np.copy(image), labels)
    return (labelled_img, heatmap)





def initialize_model():
    import os
    import pickle
    if os.path.isfile(PICKLE_FILE):
        global svc
        global X_scaler
        # load the classifier later
        print("Loading saved model from pickle data")
        file = open(PICKLE_FILE, "rb")
        svc, X_scaler = pickle.load(file)
    else:
        print("No saved model found .. run training")
        svc, X_scaler = train_data_set_and_generate_model()



##################################################
def process_jpeg_image(orig_image):

    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    image = orig_image.astype(np.float32)/255

    hot_windows_accum=multi_sliding_windows(image)

    # box_list = pickle.load(open("bbox_pickle.p", "rb"))
    window_img = draw_boxes(np.copy(orig_image), hot_windows_accum, color=COLOR_BLUE, line_thickness=4)

    box_list = hot_windows_accum

    labelled_image, heat_map_image = draw_heat_map(np.copy(orig_image), box_list)
    return window_img, heat_map_image, labelled_image


def process_video_image(orig_image):
    window_img, heat_map_image, labelled_image = process_jpeg_image(orig_image)
    return labelled_image


from moviepy.editor import VideoFileClip
def createVideo():
    video_fname = 'test_videos/project_video.mp4'
    video_outfile = 'test_videos_output/project_video.mp4'

    # video_fname = 'test_videos/test_video.mp4'
    # video_outfile = 'test_videos_output/test_video.mp4'

    clip1 = VideoFileClip(video_fname)

    processed_clip = clip1.fl_image(process_video_image)   # NOTE: this function expects color images!
    processed_clip.write_videofile(video_outfile, audio=False)


def test_image(image_name='test_images/test1.jpg'):
    image = mpimg.imread(image_name)
    window_img, heat_map_image, labelled_image = process_jpeg_image(image)
    fig = plt.figure(figsize=(12,6))
    plt.subplot(131)
    plt.imshow(window_img)
    plt.title('Detected')
    plt.subplot(132)
    plt.imshow(heat_map_image, cmap='hot')
    plt.title('Heat Map')
    plt.subplot(133)
    plt.imshow(labelled_image)
    plt.title('Labelled')
    fig.tight_layout()
    plt.show()


def test_images():
    image_names=['test_images/test1.jpg', 'test_images/test2.jpg', 'test_images/test3.jpg', 'test_images/test4.jpg'
                 ,'test_images/test5.jpg','test_images/test6.jpg']
    #image_names=['test_images/test1.jpg']

    for image_name in image_names:
        test_image(image_name)



initialize_model()
test_images()

#createVideo()

