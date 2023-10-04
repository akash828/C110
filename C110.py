# To Capture Frame
import cv2

# To process image array
import numpy as np


# import the tensorflow modules and load the model
import tensorflow as tf

model = tf.keras.models.load_model('keras_model.h5')

# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

# Infinite loop
while True:
	
	status,frame = camera.read()
	
	img = cv2.resize(frame,(224,224))
 
	frame = cv2.flip(frame , 1)
		
	test_image = np.array(img, dtype=np.float32)
	test_image = np.expand_dims(test_image, axis=0)
    
	normalised_image = test_image/225.0
		
	prediction = model.predict(normalised_image)
    
	print("Prediction : ",prediction)
		
	cv2.imshow('feed' , frame)

	code = cv2.waitKey(1)
		
	if code == 32:
		break

# release the camera from the application software
camera.release()

# close the open window
cv2.destroyAllWindows()
