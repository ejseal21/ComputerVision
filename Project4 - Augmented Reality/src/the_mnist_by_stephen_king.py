import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()

	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	blur = cv2.GaussianBlur(gray,Size(7,7),0,0)
	
	# //select grayscale values above 87 and turn them white
	ret, threshout = cv2.threshold(blur, 113, 255, THRESH_BINARY_INV)

	openMat = morphologyEx(threshout, MORPH_CLOSE, getStructuringElement(MORPH_RECT,Size(20,20)))
	
	contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	centers, numbers = [], []
	for i in range(len(contours)):
		x,y,w,h = cv2.boundingRect(contours[i])
		roi = frame[y:y+h, x:x+w]
		centers.append([x+h/2,y+w/2])
		numbers.append(roi)

	numbers = np.array(numbers)
	numbers = numbers.reshape((numbers.shape[0], 28, 28, 1))
	res = model.predict(numbers, batch_size=len(numbers))

	for i in range(len(res)):
		print("predicted:", res[i].tolist().index(max(res[i].tolist())), "at image coordinates", centers[i])

	# Display the resulting frame
	cv2.imshow('frame',frame)

	


	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()