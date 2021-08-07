import cv2 as cv 

def showImage(img, tags=""):
    cv.imshow(f"{tags}->sample image", img)
    cv.waitKey(0) # waits until a key is pressed
    cv.destroyAllWindows() # destroys the window showing image

