import json
import cv2 as cv
import tkinter as tk
import numpy as np
from multiprocessing import Array, Lock, Manager, Event

from modules.gui import gui_t
from modules.inferencer import inferencer_t
from modules.laser_cutter import laser_cutter_t
from processes.camera_process import CameraProcess

from modules.settings import ORIGINAL_IMAGE_SHAPE, SHARED_IMAGE_SHAPE, classification_image_shape, segmentation_image_shape

import ctypes

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn')

    

    # Initialize shared memory
    shared_array = Array(ctypes.c_uint8, int(np.prod(SHARED_IMAGE_SHAPE)))
    lock = Lock()
    manager = Manager()
    contour_data = manager.list()
    running_event = Event()
    running_event.set()
    quit_event = Event()
    quit_event.clear()
    
    # Initialize camera calibration
    with open("./calibration/camera_calibration.json") as f:
        calibration_dict = json.load(f)
    mtx = np.asarray(calibration_dict["mtx"])
    dist = np.asarray(calibration_dict["dist"])
    newcameramtx, _ = cv.getOptimalNewCameraMatrix(mtx, dist, (ORIGINAL_IMAGE_SHAPE[1],ORIGINAL_IMAGE_SHAPE[0]), 1, (SHARED_IMAGE_SHAPE[1],SHARED_IMAGE_SHAPE[0]))
    
    # Initialize cutter machine
    with open("./calibration/persp_matrix.json") as f:
        persp_matrix_dict = json.load(f)
    M = np.asarray(persp_matrix_dict["matrix"])
    laser_cutter = laser_cutter_t("127.0.0.1", 19840, 19841, M)

    # Initialize camera process
    cam_proc = CameraProcess(shared_array, mtx, dist, newcameramtx, lock, running_event, quit_event)
    cam_proc.start()
    
    # Initialize inferencer and GUI
    inferencer = inferencer_t("./models/class_pants_avant_arriere_chemises_v1_1.pt", classification_image_shape,
                              "./models/pants_avant_v3_1.pt", "./models/pants_arriere_v1_1.pt", "./models/chemises_v2_1.pt", segmentation_image_shape)
    root = tk.Tk()
    app = gui_t(root, inferencer, shared_array, lock, contour_data, running_event, laser_cutter)
    root.mainloop()

    # Clean up
    quit_event.set()
    cam_proc.join()