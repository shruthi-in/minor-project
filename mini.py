import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras as keras

m_new = tf.keras.models.load_model('model.h5')

img = np.ones([400,400],dtype ='uint8')*255

img[50:350,50:350]=0

wname = 'Canvas'
cv2.namedWindow(wname)
state = False
def shape(event,x,y,flags,param):
    global state
    if event == cv2.EVENT_LBUTTONDOWN:
        state=True
        cv2.circle(img,(x,y),10,(255,255,255),-1)
    if event == cv2.EVENT_LBUTTONUP:
        state = False
    if event == cv2.EVENT_MOUSEMOVE:
        if(state == True):
            cv2.circle(img,(x,y),10,(255,255,255),-1)
            print(x,y)
            
cv2.setMouseCallback(wname,shape)

while True:
    cv2.imshow(wname,img)
    key = cv2.waitkey(1)
    if key==ord('q'):
        break
    elif key==ord('c'):
        img[50:350,50:350]=0
    elif key == ord('w'):
        out = img[50:350,50:350]
        cv2.imwrite('Output.jpg',out)
    elif key == ord('p'):
        image_test = img[50:350,50:350]
        image_test_resize = cv2.resize(image_test,(28,28)).reshape(1,28,28)
        m_new.predict_classes(image_test_resize)
        print('PREDICTION = ')
cv2.destroyAllWindows()
