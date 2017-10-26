# Average heatmap over multiple frames

import numpy as np


# class to maintain a list of N recent heatmaps

class HeatMapAccumulator:

    max_images=0
    image_array = []

    def __init__(self, size=10):
        self.max_images = size


    # bounded to size max_images
    def add_heatmap_image (self, image):
        if (len(self.image_array) == self.max_images):
            self.image_array.pop(0)
        self.image_array.append(image)


    def get_summed_heat_map (self, threshold=7):
        image_sum = np.copy(self.image_array[0])
        if (len(self.image_array) == 1):
            return self.apply_threshold(image_sum, threshold)
        for i in range(1, len(self.image_array)):
            image_sum = np.add(image_sum, self.image_array[i])
        return self.apply_threshold(image_sum, threshold)


    def apply_threshold(self, heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap < threshold] = 0
        # Return thresholded map
        return heatmap

