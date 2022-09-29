#!/usr/bin/env python3

## @package SIFT_app
#  Sift transform and object identification script documentation
#
#  The following script implements object identification through the webcam using the SIFT algorithm.
#  The My_App class can be used to set up the backend of a user interface with two pushbuttons and two labels.
#  It then sets up the camera feed, enables access to an image for object identification, and then attempts to identify the 
#  object in the webcam. If the identification is strong enough, a homography is then displayed.

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi
import cv2
import sys
import numpy as np

## Object identification using the webcam (requires a User Interface)
#  
#  Class is used to connect with a pre-designed user interface with two pushbuttons and two lables.
#  The class then provides user with the ability to choose a picture from a file. The picture is displayed on the user interface.
#  Then the class connects to the webcam and using the SIFT, KNN, anf flann algorithms to detect the object and possibly display the homography.
class My_App(QtWidgets.QMainWindow):


    ## The constructor connects/subscribes to the computer's webcam and initializes the user interface.
    #  It also connects pushbuttons to two processes (one for chosing the object image and one for 
    #  starting the webcam image identification). Finally the camera intakes frames from the webcam
    #  at a set rate - calling upon SLOT_query_camera each time for image identification
    def __init__(self):
      
        # Connect to pre-defined user interface
        super(My_App, self).__init__()
        loadUi("./SIFT_app.ui", self)

        # Initialize the webcam
        self._cam_id = 0
        self._cam_fps = 2
        self._is_cam_enabled = False
        self._is_template_loaded = False

        # Connect pushbuttons to processes (image choice and webcam)
        self.browse_button.clicked.connect(self.SLOT_browse_button)
        self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

        # Connect to webcam to an Open CV object
        self._camera_device = cv2.VideoCapture(self._cam_id)
        self._camera_device.set(3, 320)
        self._camera_device.set(4, 240)

        # Timer used to trigger the camera
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.SLOT_query_camera)
        self._timer.setInterval(100 / self._cam_fps)

    ##  The convert_cv_to_pixmap converts an open CV image to a Pixmap format
    #   As a result, a pixmap image is returned, which can then be displayed in the user interface
    #   @param self the onject pointer
    #   @param cv_img an open cv image
    def convert_cv_to_pixmap(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_img.shape
        bytesPerLine = channel * width
        q_img = QtGui.QImage(cv_img.data, width, height, 
                     bytesPerLine, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_img)


    ##  The SLOT_query_function intializes the webcam frame and the template image chosen.
    #   It then applies the SIFT algortihm to the template image and the webcam image
    #   in order to determine the keypoints and the descriptors.
    #   Then it draws the keypoints onto the images. The function then computes an optimized
    #   form of the kth nearest neighbors algorithm on the descriptors.
    #   A threshold a set to determine points that have very similar descriptors (small distance).
    #   If there are enough mathching points, a homography will be formed using the homography, perspective transform,
    #   and polylines functions. If there are not enough matches, the webcam and template images will be displayed with lines between matching points
    #   @param self the onject pointer
    def SLOT_query_camera(self):
        
        # Initialize grayscaled webcam frame and template image
        ret, frame = self._camera_device.read()
        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_templ = cv2.imread(self.template_path,cv2.IMREAD_GRAYSCALE)

        # Conduct SIFt algorithm on the template image and the webcam image, then draw the keypoints on both images
        sift = cv2.xfeatures2d.SIFT_create()
        keyp_img_templ, desc_img_templ = sift.detectAndCompute(img_templ, None)
        keyp_gframe, desc_gframe = sift.detectAndCompute(grayframe, None)
        img_templ = cv2.drawKeypoints(img_templ, keyp_img_templ, img_templ)
        grayframe = cv2.drawKeypoints(grayframe, keyp_gframe, grayframe)

        # Setup and conduct k-nearest Neighbor algorithm on the descriptors of each image using the FLANN library for quick computation
        index_params = dict(algorithm = 0, trees = 5)
        search_params = dict()
        flann  = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc_img_templ, desc_gframe, k=2)

        # Select the matches that are the most similar (distances are under a threshold)
        # The threshold is set by the 0.65 multiplying the n distance. Increasing the 0.65 leads to increased matching and false positives
        poi =  []
        for m, n in matches:
            if m.distance < 0.65*n.distance:
                poi.append(m)

        # If there are enough matches, then create the homography, otherwise displays the two images with lines between the matches
        # The number 4 can be increased to for more rigidity in homography detection
        if len(poi) > 4:

            # Use the keypoints to create a mapping of the image using the homography function with RANSAC algorithm
            query_pts = np.float32([keyp_img_templ[m.queryIdx].pt for m in poi]).reshape(-1,1,2)
            train_pts = np.float32([keyp_gframe[m.trainIdx].pt for m in poi]).reshape(-1,1,2)
            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()

            # Use a perspective transform to map the contours to the webcame image - this accounts for depth 
            h, w, c = img_templ.shape
            pts = np.float32([[0,0], [0, h], [w,h], [w,0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)

            # Create the homography box using the polylines function
            homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)

            #Display the Homography
            pixmap = self.convert_cv_to_pixmap(homography)
            self.live_image_label.setPixmap(pixmap)
        else:
            # Display the mathcing points
            match_img = cv2.drawMatches(img_templ, keyp_img_templ, grayframe, keyp_gframe, poi, grayframe)
            pixmap = self.convert_cv_to_pixmap(match_img)
            self.live_image_label.setPixmap(pixmap)

            
    ## The SLOT_browse_button function opens a dialogue when the pushbutton is pressed.
    #  This dialogue enables the user to choose an image. Using the path of the image, the 
    #  function intakes the image, converts it to a pixmap format and then displays the picture in the
    #  user interface
    #  @param self the onject pointer
    def SLOT_browse_button(self):
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        
        # Intake path of image selected by user
        if dlg.exec_():
            self.template_path = dlg.selectedFiles()[0]

        # Convert to pixmap and display image in user interface
        pixmap = QtGui.QPixmap(self.template_path)
        self.template_label.setPixmap(pixmap)
        print("Loaded template image file: " + self.template_path)

    ## The SLOT_toggle_camera starts and stops the timer for camera feed intake
    #  It also checks the pushbutton to see if has been switched to Enabled or Disabled
    #  @param self the onject pointer
    def SLOT_toggle_camera(self):
        if self._is_cam_enabled:
            self._timer.stop()
            self._is_cam_enabled = False
            self.toggle_cam_button.setText("&Enable camera")
        else:
            self._timer.start()
            self._is_cam_enabled = True
            self.toggle_cam_button.setText("&Disable camera")


# Setup a myApp object and execute myApp along with user interface when invoked
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myApp = My_App()
    myApp.show()
    sys.exit(app.exec_())