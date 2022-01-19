import cv2
import numpy as np

def merge(init_frame, image):
    final_image = cv2.addWeighted(init_frame, 0.8, image, 1, 0.0)
    return final_image


def draw_line(masked_image,lines):
    lines_img = np.zeros((masked_image.shape[0], masked_image.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(lines_img,(x1,y1),(x2,y2),(255,0,0), thickness=3)
    return lines_img

def region(image):  # logic function
    mask = np.zeros_like(image) # masking the image for the particular region
    height=image.shape[0] # returns height of the image
    width=image.shape[1] # returns width of the image
    reg_list = np.array([[0,height],[width*0.45, height*0.45],[width,height]],np.int32) # reg_list array of the selected coordinates of the interested region
    reg_list.reshape((-1,1,2))  # reshaping the array to be able to read by the functions which reads a array in particular defined shape
    cv2.fillPoly(mask,[reg_list],255) # a function which will fill the region of the mask with the white(255) color.
    masked_image = cv2.bitwise_and(mask,image) # On bitwise and with image will only give the intersecting borders in the interested region
    lines = cv2.HoughLinesP(masked_image, rho=2,theta=np.pi/180, threshold=50,lines=np.array([]),minLineLength=40,maxLineGap=150) # it's a complex algo go about it can't really tell much in comments
    image_with_lines = draw_line(masked_image,lines)
    return image_with_lines

def fun(image):
    init_frame = image;
    image= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=cv2.GaussianBlur(image,(7,7),cv2.BORDER_DEFAULT)
    image=cv2.Canny(image,10,50)
    image=region(image)
    image = merge(init_frame,image)
    return image

video = cv2.VideoCapture('test_video.mp4')  # capture the video
open = video.isOpened()   # if it is captured successfully

if(open):
    while(video.isOpened()): # while the frames are running
        it,frame=video.read() # read frames
        if(not it): # if frames ends
            break
        frame=fun(frame) # main logic function
        cv2.imshow('karan1',frame) # output the final frame
        cv2.waitKey(1) # for speed/sec of a frame

video.release()  # releasing the instance
cv2.destroyAllWindows() # releasing all windows
