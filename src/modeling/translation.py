import math
import numpy as np

class translation:
    def __init__(self, lindex, rindex, horizontal_scale=1.0, vertical_scale=1.0, horizontal_transition=0.0, vertical_transition=0.0, xaxis=None, relative_length=None):
        self.lindex = lindex
        self.rindex = rindex
        self.horizontal_scale = horizontal_scale
        self.vertical_scale = vertical_scale
        self.horizontal_transition = horizontal_transition
        self.vertical_transition = vertical_transition
        self.xaxis=xaxis
        self.relative_length=relative_length

    def translate(self, clique, all):
        if not(self.xaxis is None):
            a = clique[self.xaxis[0]]
            b = clique[self.xaxis[1]]
            pitch = math.atan2(b[1] - a[1], b[0] - a[0])
            mtx_inv = np.array([[math.cos(-pitch), -math.sin(-pitch)],
                    [math.sin(-pitch), math.cos(-pitch)]])
            mtx = np.array([[math.cos(pitch), -math.sin(pitch)],
                            [math.sin(pitch), math.cos(pitch)]])
        if self.relative_length is None:
            l = 1.0 
        else: 
            x1, y1 = all[self.relative_length[0]]
            x2, y2 = all[self.relative_length[1]]
            l = math.sqrt((x1-x2)**2 + (y1-y2)**2)
        cx, cy = np.average(clique, axis=0)
        clique_ = clique - np.array([[cx, cy]])
        if not(self.xaxis is None):
            clique_ = clique_ @ mtx
        clique_[:, 0] = clique_[:, 0] * self.horizontal_scale + self.horizontal_transition * l
        clique_[:, 1] = clique_[:, 1] * self.vertical_scale  + self.vertical_transition * l
        if not(self.xaxis is None):
            clique_ = clique_ @ mtx_inv
        return np.array([[cx, cy]]) + clique_