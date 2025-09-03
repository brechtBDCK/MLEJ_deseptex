# Goal
Software for the demonstrator at centexbel.
Connects to the camera, and shows the live feed. When pressing Snap, it takes an image, runs a classification model to distinguish between "pantalon avant", "pantalon arrière" and "chemise".
Then it runs the corresponding segmentation model, post-processes the output to define the cutting lines, and shows the cutting lines.
By pressing 'edit' you can drag to adjust the polygons, press 'backspace' to remove points, press 'delete' to remove polygons, and press 'n' to add polygons.
When pressing finish, the polygons get converted to an .svg and sent to the laser cutter.

# Data
The test images and the models are located in "mechatronica\p_projects\conventions\DeSepTex\data_and_models_for_application"