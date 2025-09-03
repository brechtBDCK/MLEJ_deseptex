import cv2 as cv
import time
from multiprocessing import Process
import numpy as np
from multiprocessing.synchronize import Event as MpEvent
from modules.settings import TESTING, ORIGINAL_IMAGE_SHAPE, SHARED_IMAGE_SHAPE

if not TESTING:
    from arena_api.system import system # type: ignore
    from arena_api.buffer import BufferFactory # type: ignore
    from arena_api.__future__.save import Writer # type: ignore
    from arena_api.enums import PixelFormat # type: ignore

import ctypes
import time





class CameraProcess(Process):
    def __init__(self, shared_array, mtx, dist, newcameramtx, lock, running_flag:MpEvent, quit_event:MpEvent):
        super().__init__()
        self.shared_array = shared_array
        self.mtx = mtx
        self.dist = dist
        self.newcameramtx = newcameramtx
        self.lock = lock
        self.running_flag = running_flag
        self.quit_event = quit_event
        
        # setup camera
        if not TESTING:
            self.fps = 0.6
            self.width = ORIGINAL_IMAGE_SHAPE[1]
            self.height = ORIGINAL_IMAGE_SHAPE[0]
            self.timeout = 2000
            self.pixelformat_string = 'RGB8'
            self.pixel_format = PixelFormat.BGR8
            self.exposure_time = 600000.0
            self.num_channels = ORIGINAL_IMAGE_SHAPE[2]

            # open camera
            self.startup()
            self.setup()
            

    def run(self):
        index = 0
        
        if not TESTING:
            self.device.start_stream()
            print("Camera stream started")
        
        while not self.quit_event.is_set():
            if self.running_flag.is_set():
                
                if not TESTING:
                    # get image from camera
                    buffer = self.device.get_buffer()

                    item = BufferFactory.copy(buffer)
                    self.device.requeue_buffer(buffer)

                    buffer_bytes_per_pixel = int(len(item.data)/(item.width * item.height))
                    array = (ctypes.c_ubyte * self.num_channels * item.width * item.height).from_address(ctypes.addressof(item.pbytes))
                    image = np.ndarray(buffer=array, dtype=np.uint8, shape=(item.height, item.width, buffer_bytes_per_pixel))
                    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                    
                    # undistort and resize image
                    image_resized = cv.undistort(image, self.mtx, self.dist, None, self.newcameramtx)
                    
                    # copy image to shared memory
                    with self.lock:
                        np_array = np.frombuffer(self.shared_array.get_obj(), dtype=np.uint8).reshape(SHARED_IMAGE_SHAPE)
                        np.copyto(np_array, image_resized)
                        
                    # Destroy the copied item to prevent memory leaks
                    BufferFactory.destroy(item)
                
                else:
                    # cycle three test images
                    if index % 3 == 0:
                        frame = cv.imread("./data/test_chemise.png")
                    elif index % 3 == 1:
                        frame = cv.imread("./data/test_pant_avant.png")
                    elif index % 3 == 2:
                        frame = cv.imread("./data/test_pant_arriere.png")
                        
                    # undistort image
                    frame_resized = cv.undistort(frame, self.mtx, self.dist, None, self.newcameramtx)
                    
                    # put image in shared memory
                    with self.lock:
                        np_array = np.frombuffer(self.shared_array.get_obj(), dtype=np.uint8).reshape(SHARED_IMAGE_SHAPE)
                        np.copyto(np_array, frame_resized)
                    time.sleep(3)
                
                index += 1
            
            
            
            # running flag is not set, sleep for a while
            else:
                time.sleep(1)
                
            
        # close camera
        if not TESTING:
            self.device.stop_stream()
            system.destroy_device()
                
                
    def startup(self):
        devices = self.create_devices_with_tries()
        self.device = system.select_device(devices)
        self.nodemap = self.device.nodemap


    def create_devices_with_tries(self):
        '''
        Waits for the user to connect a device before
            raising an exception if it fails
        '''
        tries = 0
        tries_max = 6
        sleep_time_secs = 10
        devices = None
        while tries < tries_max:  # Wait for device for 60 seconds
            devices = system.create_device()
            if not devices:
                print(
                    f'Try {tries+1} of {tries_max}: waiting for {sleep_time_secs} '
                    f'secs for a device to be connected!')
                for sec_count in range(sleep_time_secs):
                    time.sleep(1)
                    print(f'{sec_count + 1 } seconds passed ',
                        '.' * sec_count, end='\r')
                tries += 1
            else:
                return devices
        else:
            raise Exception(f'No device found! Please connect a device and run '
                            f'the example again.')


    def setup(self):
        nodes = self.nodemap.get_node(['Width', 'Height', 'PixelFormat', 'AcquisitionFrameRateEnable', 'AcquisitionFrameRate', 'ExposureAuto', 'ExposureTime'])
        
        nodes['Width'].value = self.width
        nodes['Height'].value = self.height
        nodes['PixelFormat'].value = self.pixelformat_string
        
        # Get device stream nodemap
        tl_stream_nodemap = self.device.tl_stream_nodemap
        tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
        tl_stream_nodemap['StreamPacketResendEnable'].value = True

        # set framerate
        nodes['AcquisitionFrameRateEnable'].value = True
        nodes['AcquisitionFrameRate'].value = self.fps
        
        # set exposure time
        nodes['ExposureAuto'].value = 'Off'
        nodes['ExposureTime'].value = self.exposure_time