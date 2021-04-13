#!/usr/bin/env python
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import sys
import cv2.aruco as aruco
import cv2, PIL
import matplotlib.pyplot as plt
import matplotlib as mpl

bridge = CvBridge()

class params:
    threshConsant = 7
    threshWinSizeMax = 23
    threshWinSizeMin = 3
    threshWinSizeStep = 10
    accuracyRate = 0.02
    minAreaRate = 0.03
    maxAreaRate = 4
    minCornerDisRate = 2.5
    minMarkerDisRate = 1.5

def remove_close_candidates(candidates):
    newCandidates = list()

    for i in range(len(candidates)):
        for j in range(len(candidates)):
            
            if i == j:
                continue

            minPerimeter = min(cv2.arcLength(candidates[i], True), cv2.arcLength(candidates[j], True))

            
            for fc in range(4):
                disSq = 0
                for c in range(4):
                    modC = (fc + c) % 4
                    dx = candidates[j][c][0][0] - candidates[i][modC][0][0]
                    dy = candidates[j][c][0][1] - candidates[i][modC][0][1]
                    disSq += dx * dx + dy * dy
                disSq /= 4

                minDisPixels = minPerimeter * params.minMarkerDisRate

                if disSq < minDisPixels * minDisPixels:
                    if cv2.contourArea(candidates[i]) > cv2.contourArea(candidates[j]):
                        newCandidates.append(candidates[i])
                    else:
                        newCandidates.append(candidates[j])

    
    if len(newCandidates):
        return newCandidates
    else:
        return candidates


def has_close_corners(candidate):
    minDisSq = float("inf")

    for i in range(len(candidate)):
        dx = candidate[i][0][0] - candidate[(i+1)%4][0][0]
        dy = candidate[i][0][1] - candidate[(i+1)%4][0][1]
        dsq = dx * dx + dy * dy
        minDisSq = min(minDisSq, dsq)

    minDisPixel = candidate.size * params.minCornerDisRate
    if minDisSq < minDisPixel * minDisPixel:
        return True
    else:
        return False


def detect_candidates(grayImg):
    th = cv2.adaptiveThreshold(grayImg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, params.threshConsant)
    cnts = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2]

    
    candidates = list()
    for c in cnts:
        
        maxSize = int(max(gray.shape) * params.maxAreaRate)
        minSize = int(max(gray.shape) * params.minAreaRate)
        if c.size > maxSize or c.size < minSize:
            continue

        approxCurve = cv2.approxPolyDP(c, len(c) * params.accuracyRate, True)
        
        if len(approxCurve) is not 4 or cv2.isContourConvex(approxCurve) is False:
            continue

        
        if has_close_corners(approxCurve):
            continue
        
        candidates.append(approxCurve)

    return candidates



def read_rgb_image(image_name, show):
    rgb_image = cv2.imread(image_name)
    if show:
        cv2.namedWindow("RGB Image", ())
        cv2.imshow("RGB Image", rgb_image)
    return rgb_image


def color_filter(rgb_image, lower_bound_color, upper_bound_color):
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    cv2.namedWindow("HSV Image", cv2.WINDOW_NORMAL)
    cv2.imshow("HSV Image", hsv)
    # find lower and upper bounds
    yellowLower = (120, 150, 100)
    yellowUpper = (150, 255, 255)

    mask = cv2.inRange(hsv, lower_bound_color, upper_bound_color)

    return mask

def getContours(binary_image):      
    #_, contours, hierarchy = cv2.findContours(binary_image, 
    #                                          cv2.RETR_TREE, 
    #                                           cv2.CHAIN_APPROX_SIMPLE)
    _, contours, hierarchy = cv2.findContours(binary_image.copy(), 
                                            cv2.RETR_EXTERNAL,
	                                        cv2.CHAIN_APPROX_SIMPLE)
    return contours    

def draw_contours(image, contours, image_name):
    index = -1 #means all contours
    thickness = 4
    color = (255, 0, 255)
    cv2.drawContours(image, contours, index, color, thickness)
    cv2.imshow(image_name, image)

def process_contours(binary_image, rgb_image, contours):
    black_image = np.zeros([binary_image.shape[0], binary_image.shape[1], 3]) 
    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        ((x,y), radius) = cv2.minEnclosingCircle(c)
        if(area > 0):
            cv2.drawContours(rgb_image, [c], -1, (150, 250, 150), 1)
            cv2.drawContours(black_image, [c], -1, (150, 250, 150), 1)
            cx, cy = get_contour_center(c)
            cv2.circle(rgb_image, (cx, cy), (int)(radius), (0,0,255), 1)
            cv2.circle(black_image, (cx, cy), (int)(radius), (0,0,255), 1)
            cv2.circle(black_image, (cx, cy), 5, (0,0,255), 1)
            #print("Area: {}, Perimeter: {}".format(area, perimeter))
    print("Number of BOLITAS: {}".format(len(contours)))
    cv2.namedWindow("RGB Image Contours", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Black Image Contours", cv2.WINDOW_NORMAL)
    cv2.imshow("RGB Image Contours", rgb_image)
    cv2.imshow("Black Image Contours", black_image)

def get_contour_center(contour):
    M = cv2.moments(contour)
    cx = -1
    cy = -1
    if(M['m00'] != 0):
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    return cx, cy

def image_callback(ros_image):
    print("got an image")
    global bridge

    try:
        cv_image = bridge.imgmsg_to_cv2(ros_image, "bgr8") #cv image es la vaina
    except CvBridgeError as e:
        print(e)

    frame = cv_image
    
    frame = cv2.GaussianBlur(frame, (5,5), 0)
    global gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    candidates = detect_candidates(gray)

    if len(candidates) > 0:
        candidates = remove_close_candidates(candidates)

    cv2.drawContours(frame, candidates, -1, (0, 255, 0), 3)
    cv2.imshow('frame', frame)
    cv2.waitKey(2)


    
def main(args):
    rospy.init_node("only_frame", anonymous = True)
    # for turtlebot3 waffle
    # image_topic = "/camera/rgb/image_raw/compressed"
    # for usb cam
    # image_topic= "/usb_cam/image_raw"
    image_sub = rospy.Subscriber("/zed2/left/image_rect_color", Image, image_callback)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting Down...")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)