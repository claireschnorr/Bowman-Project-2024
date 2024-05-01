# importing necessary libraries
import cv2
from matplotlib import pyplot as plt
import numpy as np 


# read a cracked sample image
for n in range(1, 67):
    imageNum = str(n)
    inputTitle = "Bowman-Input-Set/Picture "+imageNum+".jpg"
    img = cv2.imread(inputTitle)
    #picture 40 picture 55

    # Convert into gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # # # Image processing ( smoothing )
    # # # Averaging
    blur = cv2.blur(gray,(15,15)) #higher the numerical value the stronger out of focus the image


    # # # # Apply logarithmic transform

    img_log = (np.log(blur+1)/(np.log(1+np.max(blur))))*255  #increases brightness of the image

    # # # # Specify the data type
    img_log = np.array(img_log,dtype=np.uint8)


    # # # Image smoothing: bilateral filter

    bilateral = cv2.bilateralFilter(blur, 1, 75, 75) #5 75 75
    #featuredImg = bilateral


    # # # Canny Edge Detection
    edges = cv2.Canny(bilateral,6,20) #image for presentation used 6 20

    # # # Morphological Closing Operator
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    #featuredImg = closing


    # # # Create feature detecting method
    # # sift = cv2.xfeatures2d.SIFT_create()
    # # surf = cv2.xfeatures2d.SURF_create()
    orb = cv2.ORB_create(nfeatures=1500)

    # # # Make featured Image
    keypoints, descriptors = orb.detectAndCompute(closing, None)
    featuredImg = cv2.drawKeypoints(closing, keypoints, None)

    # Create an output image
    outputName = "Bowman-Output-Set/CrackDetected-"+imageNum+".jpg"
    cv2.imwrite(outputName, featuredImg)

    # # Use plot to show original and output image
    # plt.subplot(121),plt.imshow(img)
    # plt.title('Original'),plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(featuredImg,cmap='gray')
    # plt.title('Output Image'),plt.xticks([]), plt.yticks([])
    # plt.show()
