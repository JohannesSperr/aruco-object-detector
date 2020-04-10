import os
import math
import csv
import numpy as np
import pandas as pd
import cv2
import cv2.aruco as aruco


class Detector():
    radiant_factor = 180/math.pi
    factor_x = 1.00
    factor_y = 1.00
    factor_z = 1.00
    dist_coeffs = np.zeros((4,1))  # Assuming no lence curvature
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)  # Load the aruco marker dictionary
    parameters = aruco.DetectorParameters_create()

    def __init__(self, camera_matrix = None):
        self.camera_matrix = camera_matrix
        self.data = None
        self.attitude_angles = None
        self.identified_marker_objects = None

    def read_data(self):
        self.data = pd.read_csv('data.csv', delimiter=';')        

    def create_camera_matrix(self, frame):
        focal_length = frame.shape[1]
        center = (frame.shape[1]/2, frame.shape[0]/2)
        self.camera_matrix = np.array([[focal_length, 0, center[0]], 
                                        [0, focal_length, center[1]], 
                                        [0, 0, 1]], dtype = "double")

    def identify_marker_objects(self, frame, ids, rvec, tvec):
        self.identified_marker_objects = []
        for counter, value in enumerate(ids):
            aruco.drawAxis(frame, self.camera_matrix, Detector.dist_coeffs, rvec[counter], tvec[counter], 0.06)
            tvec[counter] = tvec[counter] * (Detector.factor_x, Detector.factor_y, Detector.factor_z)
            rot = cv2.Rodrigues(rvec[counter])
            self.calculate_attitude_angles(rot[0])

            if (value in self.data.marker_id.values) == True:
                data_current_marker = self.data.loc[value, :].values.tolist()
            else:
                data_current_marker = [None, None]

            self.identified_marker_objects.append([ids[counter], data_current_marker, tvec[counter], 
                                                    rvec[counter] * Detector.radiant_factor, rot[0], 
                                                    self.attitude_angles]) 
           
    def calculate_attitude_angles(self, rot):
        cos_beta = math.sqrt(rot[2,1] * rot[2,1] + rot[2,2] * rot[2,2])
        validity = cos_beta < 1e-6
        if not validity:
            angle_yaw = math.atan2(rot[1,0], rot[0,0])    
            angle_pitch = math.atan2(-rot[2,0], cos_beta) 
            angle_roll = math.atan2(rot[2,1], rot[2,2])    
        else:
            angle_yaw = math.atan2(rot[1,0], rot[0,0])    
            angle_pitch = math.atan2(-rot[2,0], cos_beta)
            angle_roll = 0

        self.attitude_angles = np.array([angle_yaw, angle_pitch, angle_roll]) * Detector.radiant_factor

    def flatten_list_ids(self, items):
        new_list = []
        for item in items:
            new_list.append(item[0])
        return new_list

    def flatten_list_objects(self, items):
        new_list = []
        for item in items:
            if item[1] is not None:
                new_list.append(item[1][1])
            else:
                new_list.append(None)
        return new_list

    def display_data(self, ids):
        font = cv2.FONT_HERSHEY_SIMPLEX
        identified_objects = self.flatten_list_objects(self.identified_marker_objects)
        identified_objects = 'objects: ' + str(identified_objects)
        display_text_marker = 'marker_ids: ' + str(ids)
        cv2.putText(frame, identified_objects, (10,430), font, 0.45, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(frame, display_text_marker, (10,450), font, 0.45, (255,255,255), 1, cv2.LINE_AA)


if __name__ == "__main__":
    detector = Detector()
    detector.read_data()

    cap = cv2.VideoCapture(0)

    while(True): 
        ret, frame = cap.read()
        detector.create_camera_matrix(frame)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, Detector.aruco_dict, parameters=Detector.parameters)

        if corners != []:
            rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, 0.05, detector.camera_matrix, Detector.dist_coeffs)
            ids = detector.flatten_list_ids(ids)
            detector.identify_marker_objects(frame, ids, rvec, tvec)
            detector.display_data(ids)
        else:
            pass

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 0xFF == ord('c'):
            cv2.imwrite('screenshot.png',frame)
        else:
            pass

    cap.release()
    cv2.destroyAllWindows()
