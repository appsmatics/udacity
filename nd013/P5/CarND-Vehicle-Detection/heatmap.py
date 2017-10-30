import numpy as np

# class to maintain a list and accumulation of N recent heatmaps
# A threshold can also be applied to the accumulated heatmap
# i.e. heatmap[:,:] < threshold = 0

class HeatMapAccumulator:

    max_hetamaps=0
    threshold=0
    heatmap_array = []
    heatmap_sum = None

    def __init__(self, size=10, threshold=5):
        self.max_hetamaps = size
        self.threshold = threshold
        self.heatmap_array = []
        self.heatmap_sum = None


    # bounded to size max_hetamaps
    def add_heatmap (self, heatmap):

        if (len(self.heatmap_array) == 0):
            self.heatmap_sum = np.copy(heatmap)
            self.heatmap_array.append(heatmap)
            return

        if (len(self.heatmap_array) == self.max_hetamaps):
            heat = self.heatmap_array.pop(0)
            self.heatmap_sum = np.subtract(self.heatmap_sum, heat)

        self.heatmap_sum = np.add(self.heatmap_sum, heatmap)
        self.heatmap_array.append(heatmap)


    def get_summed_heat_map_after_threshold (self):
        heatmap = np.copy(self.heatmap_sum)
        return self.apply_threshold(heatmap, self.threshold)


    def apply_threshold(self, heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap < threshold] = 0
        # Return thresholded map
        return heatmap

