import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import cv2 as cv
from multiprocessing.synchronize import Event as MpEvent, Lock as MpLock
from multiprocessing.sharedctypes import SynchronizedArray
from multiprocessing.managers import ListProxy

from modules.inferencer import inferencer_t
from modules.laser_cutter import laser_cutter_t
from modules.settings import TESTING, ORIGINAL_IMAGE_SHAPE, SHARED_IMAGE_SHAPE, GUI_IMAGE_SHAPE

class gui_t:
    def __init__(self, root:tk.Tk,
                 inferencer:inferencer_t,
                 shared_array:SynchronizedArray,
                 lock:MpLock,
                 contour_data:ListProxy,
                 running_event:MpEvent,
                 laser_cutter:laser_cutter_t):
        
        self.root = root
        self.inferencer = inferencer
        self.shared_array = shared_array
        self.lock = lock
        self.contour_data = contour_data
        self.running_event = running_event
        self.laser_cutter = laser_cutter
        
        self.update_job = None
        self.edit_mode = False
        self.refresh_time = 100  # milliseconds

        self.root.title("Image Viewer")
        self.root.geometry("1000x700")

        self.top_frame = ttk.Frame(self.root)
        self.top_frame.pack(side="top", fill="x")

        self.quit_button = ttk.Button(self.top_frame, text="Quit", command=self.root.quit)
        self.quit_button.pack(side="right")

        self.snap_button = ttk.Button(self.top_frame, text="Snap", command=self.toggle_running)
        self.snap_button.pack(side="left")

        self.edit_button = ttk.Button(self.top_frame, text="Edit", command=self.toggle_edit, state="disabled")
        self.edit_button.pack(side="left")

        self.finish_button = ttk.Button(self.top_frame, text="Finish", command=self.send_to_laser_cutter)
        self.finish_button.pack(side="left")

        self.canvas_frame = ttk.Frame(self.root)
        self.canvas_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg="black")
        self.canvas.pack(fill="both", expand=True)
        
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        self.point_radius = 5
        self.hover_circle = None
        self.dragging_point = None  # (polygon_index, point_index)

        self.update_content()
        
        self.root.bind("<BackSpace>", self.on_backspace)
        self.root.bind("<Delete>", self.on_delete)
        self.root.bind("n", self.on_n)

    def toggle_running(self):
        if self.running_event.is_set():
            # get image from shared memory
            with self.lock:
                img_np = np.frombuffer(self.shared_array.get_obj(), dtype=np.uint8).reshape(SHARED_IMAGE_SHAPE).copy() #type: ignore
            
            # do inference
            self.contour_data = self.inferencer.run_inference_with_postproc(img_np)
            
            # resize contour data to match the GUI image shape
            for i in range(len(self.contour_data)):
                self.contour_data[i] = np.array(self.contour_data[i]) * (GUI_IMAGE_SHAPE[1] / SHARED_IMAGE_SHAPE[1], GUI_IMAGE_SHAPE[0] / SHARED_IMAGE_SHAPE[0]) #type: ignore
                self.contour_data[i] = self.contour_data[i].astype(np.int32)

            # draw resulting contours on the canvas
            self.update_polygons()
            
            # update other variables
            self.running_event.clear()
            self.edit_button.config(state="normal")
            self.snap_button.config(text="Resume")
            if self.update_job is not None:
                self.root.after_cancel(self.update_job)
                self.update_job = None
        else:
            self.running_event.set()
            self.edit_button.config(state="disabled")
            self.edit_mode = False
            self.snap_button.config(text="Snap")
            self.update_content()


    def toggle_edit(self):
        self.edit_mode = not self.edit_mode


    def update_content(self):
        self.update_image()
        if self.running_event.is_set():
            self.update_job = self.root.after(self.refresh_time, self.update_content)
        else:
            self.update_job = None
    
    
    def update_image(self):
        with self.lock:
            img_np = np.frombuffer(self.shared_array.get_obj(), dtype=np.uint8).reshape(SHARED_IMAGE_SHAPE).copy() #type: ignore
        resized_img_np = cv.resize(img_np, (GUI_IMAGE_SHAPE[1],GUI_IMAGE_SHAPE[0]))
        img = Image.fromarray(cv.cvtColor(resized_img_np, cv.COLOR_BGR2RGB))
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor='nw', image=self.tk_img)
    
    
    def update_polygons(self):
        self.canvas.delete("polygon")
        for cnt in self.contour_data:
            if len(cnt) > 2:
                flat = [coord for point in cnt for coord in point]
                self.canvas.create_polygon(flat, outline='red', fill='', width=2, tags="polygon")
    
    
    def on_mouse_move(self, event):
        if not self.edit_mode:
            return
        
        x, y = event.x, event.y
        closest = self.find_closest_point(x, y)

        if closest:
            poly_idx, pt_idx = closest
            px, py = self.contour_data[poly_idx][pt_idx]
            self.show_hover_circle(px, py)
        else:
            self.hide_hover_circle()
    
    
    def on_mouse_down(self, event):
        if not self.edit_mode:
            return
        
        closest = self.find_closest_point(event.x, event.y)
        if closest:
            self.dragging_point = closest
            poly_idx, pt_idx = closest
            px, py = self.contour_data[poly_idx][pt_idx]
            self.show_hover_circle(px, py)
    
    
    def on_mouse_drag(self, event):
        if not self.edit_mode or self.dragging_point is None:
            return
        poly_idx, pt_idx = self.dragging_point
        polygon = self.contour_data[poly_idx]
        polygon[pt_idx] = [event.x, event.y]
        self.contour_data[poly_idx] = polygon # assignment necessary on this level to update the manager list
        self.update_polygons()
        self.show_hover_circle(event.x, event.y)


    def on_mouse_up(self, event):
        self.dragging_point = None
        self.hide_hover_circle()
    
    
    def show_hover_circle(self, x, y):
        self.hide_hover_circle()
        self.hover_circle = self.canvas.create_oval(
            x - self.point_radius, y - self.point_radius,
            x + self.point_radius, y + self.point_radius,
            outline='blue', width=2)


    def hide_hover_circle(self):
        if self.hover_circle is not None:
            self.canvas.delete(self.hover_circle)
            self.hover_circle = None
            
            
    def find_closest_point(self, x, y, threshold=10):
        for poly_idx, contour in enumerate(self.contour_data):
            for pt_idx, (px, py) in enumerate(contour):
                if (x - px)**2 + (y - py)**2 <= threshold**2:
                    return (poly_idx, pt_idx)
        return None

    
    def on_backspace(self, event):  
        if not self.edit_mode:
            return

        x = self.canvas.winfo_pointerx() - self.canvas.winfo_rootx()
        y = self.canvas.winfo_pointery() - self.canvas.winfo_rooty()
        
        closest = self.find_closest_point(x, y)
        if closest:
            poly_idx, pt_idx = closest
            polygon = self.contour_data[poly_idx]

            if len(polygon) > 3:  # Ensure it remains a valid polygon
                # check if polygon is a list
                if not isinstance(polygon, list):
                    polygon_list = polygon.tolist()
                else:
                    polygon_list = polygon
                del polygon_list[pt_idx]
                self.contour_data[poly_idx] = np.array(polygon_list, dtype=np.int32) # type: ignore
                self.update_polygons()
                self.hide_hover_circle()

    
    def on_delete(self, event):
        if not self.edit_mode:
            return

        x = self.canvas.winfo_pointerx() - self.canvas.winfo_rootx()
        y = self.canvas.winfo_pointery() - self.canvas.winfo_rooty()

        closest = self.find_closest_point(x, y)
        if closest:
            poly_idx, _ = closest
            del self.contour_data[poly_idx]
            self.update_polygons()
            self.hide_hover_circle()
    
    
    def on_n(self, event):
        if not self.edit_mode:
            return
            
        size = 10
        x = self.canvas.winfo_pointerx() - self.canvas.winfo_rootx()
        y = self.canvas.winfo_pointery() - self.canvas.winfo_rooty()
        
        new_contour = np.array([[x - size, y - size],
                                [x + size, y - size],
                                [x + size, y + size],
                                [x - size, y + size]], dtype=np.int32)
        self.contour_data.append(new_contour) # type: ignore
        self.update_polygons()
    
    
    def send_to_laser_cutter(self):
        resized_contour_data = []
        for i in range(len(self.contour_data)):
            new_contour = np.array(self.contour_data[i]) * (ORIGINAL_IMAGE_SHAPE[1] / GUI_IMAGE_SHAPE[1], ORIGINAL_IMAGE_SHAPE[0] / GUI_IMAGE_SHAPE[0])
            new_contour = np.concatenate((new_contour, new_contour[:1]), axis=0)
            resized_contour_data.append(new_contour)
            resized_contour_data[i] = resized_contour_data[i].astype(np.int32)

        self.laser_cutter.prepare_svg(resized_contour_data)
        if not TESTING:
            self.laser_cutter.start_cutter()
            self.laser_cutter.send_svg_to_cutter()