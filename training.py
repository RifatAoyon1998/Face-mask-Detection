#
#
# # import cv2
# # import numpy as np
# # from os import listdir
# # from os.path import isfile, join
# #
# # data_path='ABCD'
# # onlyfiles=[f for f in listdir(data_path) if isfile(join(data_path))]
# #
# # Training_Data, Labels = [], []
# #
# # for i, files in enumerate(onlyfiles):
# #     image_path = data_path + onlyfiles[i]
# #     images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# #     Training_Data.append(np.asarray(images, dtype=np.uint))
# #     # Labels.append(i)
# #
# # Labels = np.asarray(Labels, dtype=np.int)
# # # model = cv2.face.LBPHFaceRecognizer_create()
# #
# # model = cv2.face.LBPHFaceRecognizer_create()
# #
# # # model.train(np.asarray(Training_Data), np.asarray(Labels))
# # model.train(np.asarray(Training_Data), np.asarray(Labels))
# # print("Dataset Model Training Completed ")
#
#
#
# # import cv2,os
# # import numpy as np
# # from PIL import Image
# #
# # recognizer = cv2.createLBPHFaceRecognizer()
# # detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
# #
# # def getImagesAndLabels(path):
# #     #get the path of all the files in the folder
# #     imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
# #     #create empth face list
# #     faceSamples=[]
# #     #create empty ID list
# #     Ids=[]
# #     #now looping through all the image paths and loading the Ids and the images
# #     for imagePath in imagePaths:
# #
# #         # Updates in Code
# #         # ignore if the file does not have jpg extension :
# #         if(os.path.split(imagePath)[-1].split(".")[-1]!='jpg'):
# #             continue
# #
# #         #loading the image and converting it to gray scale
# #         pilImage=Image.open(imagePath).convert('L')
# #         #Now we are converting the PIL image into numpy array
# #         imageNp=np.array(pilImage,'uint8')
# #         #getting the Id from the image
# #         Id=int(os.path.split(imagePath)[-1].split(".")[1])
# #         # extract the face from the training image sample
# #         faces=detector.detectMultiScale(imageNp)
# #         #If a face is there then append that in the list as well as Id of it
# #         for (x,y,w,h) in faces:
# #             faceSamples.append(imageNp[y:y+h,x:x+w])
# #             Ids.append(Id)
# #     return faceSamples,Ids
# #
# #
# # faces,Ids = getImagesAndLabels('dataSet')
# # recognizer.train(faces, np.array(Ids))
# # recognizer.save('trainner/trainner.yml')
# # print("OK")
#
# #
# # import numpy as np
# # import face_recognition as fr
# # import cv2
# # import datetime
# # # import os
# # import pygame
# # import pyttsx3
# #
# # speaker = pyttsx3.init()
# # video_capture = cv2.VideoCapture(0)
# #
# # bruno_image = fr.load_image_file("1671156.jpg")
# # bruno_face_encoding = fr.face_encodings(bruno_image)[0]
# # bruno_image2 = fr.load_image_file("mark.jpg")
# # bruno_face_encoding2 = fr.face_encodings(bruno_image2)[0]
# # bruno_image3 = fr.load_image_file("elon.jpeg")
# # bruno_face_encoding3 = fr.face_encodings(bruno_image3)[0]
# # known_face_encondings = [bruno_face_encoding,bruno_face_encoding2,bruno_face_encoding3]
# # known_face_names = ["Aoyon","Mark","Elon"]
# # fn=""
# # nameList=[]
# # def mark_attendance(name):
# #     """
# #     :param name: detected face known or unknown one
# #     :return:
# #     """
# #     with open('atnd.csv', 'a') as f:
# #         if name not in nameList:
# #             nameList.append(name)
# #             date_time_string = datetime.datetime.now().strftime("%y/%m/%d %H:%M:%S")
# #             f.writelines(f'\n{name},{date_time_string}')
# #
# #
# #
# # while True:
# #     ret, frame = video_capture.read()
# #
# #     rgb_frame = frame[:, :, ::-1]
# #
# #     face_locations = fr.face_locations(rgb_frame)
# #     face_encodings = fr.face_encodings(rgb_frame, face_locations)
# #
# #     for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
# #
# #         matches = fr.compare_faces(known_face_encondings, face_encoding)
# #
# #         name = "Unknown"
# #
# #         face_distances = fr.face_distance(known_face_encondings, face_encoding)
# #
# #         best_match_index = np.argmin(face_distances)
# #         if matches[best_match_index]:
# #             name = known_face_names[best_match_index]
# #             mark_attendance(name)
# #         fn=name
# #         if name=="Unknown":
# #             pygame.init()
# #             pygame.mixer.music.load("alert.wav")
# #             pygame.mixer.music.play()
# #         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
# #
# #         cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
# #         font = cv2.FONT_HERSHEY_SIMPLEX
# #         cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
# #
# #     cv2.imshow('Webcam_facerecognition', frame)
# #     # if fn != "Unknown":
# #     #     speaker.say("Bonjour" + name + ". Comment ca va??")
# #     #
# #     #     speaker.runAndWait()
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break
# # # df=pd.read_csv('atnd.csv')
# # # writer=pd.ExcelWriter('atnd.xlsx')
# # # df.to_excel(writer,index=False)
# # # writer.save()
# # video_capture.release()
# #
# # cv2.destroyAllWindows()
#
#
#
# # import  cv2
# # import mediapipe as mp
# # import  time
# #
# # class handDetector():
# #     def __init__(self, mode=False, maxHands = 2, detectionCon = 0.7, trackCon=0.5):
# #         self.mode = mode
# #         self.maxhands = maxHands
# #         self.detectionCon = detectionCon
# #         self.trackCon = trackCon
# #         self.mphands = mp.solutions.hands
# #         self.hands = self.mphands.Hands(self.mode,
# #                                         self.maxhands,
# #                                         self.detectionCon,
# #                                         self.trackCon)
# #         self.mpdraw = mp.solutions.drawing_utils
# #
# #
# #     def findHands(self, img, draw=True):
# #          imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
# #          self.results = self.hands.process(imgRGB)
# #          # print(results.multi_hand_landmarks)
# #
# #          if self.results.multi_hand_landmarks:
# #              for handLms in self.results.multi_hand_landmarks:
# #                  if draw:
# #                     self.mpdraw.draw_landmarks(img, handLms, self.mphands.HAND_CONNECTIONS)
# #          return img
# #
# #     def findposition(self, img, handNo = 0 , draw = True):
# #
# #         lmlist = []
# #         if self.results.multi_hand_landmarks:
# #             myhand = self.results.multi_hand_landmarks[handNo]
# #
# #             for id, lm in enumerate(myhand.landmark):
# #               # print(id,lm)
# #               h, w, c = img.shape
# #               cx, cy = int(lm.x * w), int(lm.y * h)
# #               print(id, cx, cy)
# #               lmlist.append([id,cx,cy])
# #               if draw:
# #                  cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
# #
# #         return  lmlist
# #
# #
# #
# #
# #
# #
# #
# #
# # def main():
# #     pTime = 0
# #     cTime = 0
# #     cap = cv2.VideoCapture(0)
# #     detector = handDetector()
# #
# #     while True:
# #         success, img = cap.read()
# #         img = detector.findHands(img)
# #         lmlist = detector.findposition(img)
# #         if len(lmlist) != 0:
# #             print(lmlist[4])
# #
# #         cTime = time.time()
# #         fps = 1 / (cTime - pTime)
# #         pTime = cTime
# #
# #         cv2.putText(img, f'FPS : {int(fps)}', (30, 50), cv2.FONT_ITALIC, 2, (255, 0, 0), 3)
# #
# #         cv2.imshow("IMAGE", img)
# #         cv2.waitKey(1)
# #
# #
# #
# #
# #
# # if __name__ == "__main__":
# #     main()
#
#
#
# # import the necessary packages
# # import pyautogui
# # from scipy.spatial import distance as dist
# # #from imutils.video import FileVideoStream
# # #from imutils.video import VideoStream
# # from imutils import face_utils
# # #import numpy as np
# # #import argparse
# # import imutils
# # import time
# # import dlib
# # import cv2
# #
# #
# #
# # def eye_aspect_ratio(eye):
# #     # compute the euclidean distances between the two sets of
# #     # vertical eye landmarks (x, y)-coordinates
# #     A = dist.euclidean(eye[1], eye[5])
# #     B = dist.euclidean(eye[2], eye[4])
# #
# #     # compute the euclidean distance between the horizontal
# #     # eye landmark (x, y)-coordinates
# #     C = dist.euclidean(eye[0], eye[3])
# #
# #     # compute the eye aspect ratio
# #     ear = (A + B) / (2.0 * C)
# #
# #     # return the eye aspect ratio
# #     return ear
# #
# #
# # # construct the argument parse and parse the arguments
# # """
# # ap = argparse.ArgumentParser()
# # ap.add_argument("-p", "--shape-predictor", required=True,
# # 	help="path to facial landmark predictor")
# # ap.add_argument("-v", "--video", type=str, default="",
# # 	help="path to input video file")
# # args = vars(ap.parse_args())
# # """
# #
# # # shape=predictor(photo,detect[0])
# #
# # # define two constants, one for the eye aspect ratio to indicate
# # # blink and then a second constant for the number of consecutive
# # # frames the eye must be below the threshold
# # EYE_AR_THRESH = 0.3
# # EYE_AR_CONSEC_FRAMES = 3
# #
# # # initialize the frame counters and the total number of blinks
# # COUNTER = 0
# # TOTAL = 0
# #
# # # initialize dlib's face detector (HOG-based) and then create
# # # the facial landmark predictor
# # print("[INFO] loading facial landmark predictor...")
# # detector = dlib.get_frontal_face_detector()
# # # predictor = dlib.shape_predictor(args["shape_predictor"])
# # predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# # # grab the indexes of the facial landmarks for the left and
# # # right eye, respectively
# # (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
# # (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
# #
# # # start the video stream thread
# # print("[INFO] starting video stream thread...")
# # # vs = FileVideoStream(args["video"]).start()
# # # fileStream = True
# # #vs = VideoStream(src=0).start()
# # vs = cv2.VideoCapture(0)
# # # vs = VideoStream(usePiCamera=True).start()
# # fileStream = False
# # time.sleep(1.0)
# #
# # # loop over frames from the video stream
# # while True:
# #     # if this is a file video stream, then we need to check if
# #     # there any more frames left in the buffer to process
# #     ret,frame = vs.read()
# #
# #     if fileStream and not ret:
# #         break
# #
# #     # grab the frame from the threaded video file stream, resize
# #     # it, and convert it to grayscale
# #     # channels)
# #
# #     frame = imutils.resize(frame, width=450)
# #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #
# #     # detect faces in the grayscale frame
# #     rects = detector(gray, 0)
# #
# #     # loop over the face detections
# #     for rect in rects:
# #         # determine the facial landmarks for the face region, then
# #         # convert the facial landmark (x, y)-coordinates to a NumPy
# #         # array
# #         shape = predictor(gray, rect)
# #         shape = face_utils.shape_to_np(shape)
# #
# #         # extract the left and right eye coordinates, then use the
# #         # coordinates to compute the eye aspect ratio for both eyes
# #         leftEye = shape[lStart:lEnd]
# #         rightEye = shape[rStart:rEnd]
# #         leftEAR = eye_aspect_ratio(leftEye)
# #         rightEAR = eye_aspect_ratio(rightEye)
# #
# #         # average the eye aspect ratio together for both eyes
# #         ear = (leftEAR + rightEAR) / 2.0
# #
# #         # compute the convex hull for the left and right eye, then
# #         # visualize each of the eyes
# #         leftEyeHull = cv2.convexHull(leftEye)
# #         rightEyeHull = cv2.convexHull(rightEye)
# #         cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
# #         cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
# #
# #         # check to see if the eye aspect ratio is below the blink
# #         # threshold, and if so, increment the blink frame counter
# #         if ear < EYE_AR_THRESH:
# #             COUNTER += 1
# #
# #         # otherwise, the eye aspect ratio is not below the blink
# #         # threshold
# #         else:
# #             # if the eyes were closed for a sufficient number of
# #             # then increment the total number of blinks
# #             if COUNTER >= EYE_AR_CONSEC_FRAMES:
# #                 pyautogui.press('space')
# #                 TOTAL += 1
# #
# #
# #             # reset the eye frame counter
# #             COUNTER = 0
# #
# #         # draw the total number of blinks on the frame along with
# #         # the computed eye aspect ratio for the frame
# #         cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
# #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
# #         cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
# #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
# #
# #     # show the frame
# #     cv2.imshow("Frame", frame)
# #     key = cv2.waitKey(1) & 0xFF
# #
# #     # if the `q` key was pressed, break from the loop
# #     if key == ord("q"):
# #         break
# #
# # # do a bit of cleanup
# # vs.release()
# # cv2.destroyAllWindows()
#
#
# import pyautogui
# from scipy.spatial import distance as dist
# #from imutils.video import FileVideoStream
# #from imutils.video import VideoStream
# from imutils import face_utils
# #import numpy as np
# #import argparse
# import imutils
# import time
# import dlib
# import cv2
#
#
#
# def eye_aspect_ratio(eye):
#     # compute the euclidean distances between the two sets of
#     # vertical eye landmarks (x, y)-coordinates
#     A = dist.euclidean(eye[1], eye[5])
#     B = dist.euclidean(eye[2], eye[4])
#
#     # compute the euclidean distance between the horizontal
#     # eye landmark (x, y)-coordinates
#     C = dist.euclidean(eye[0], eye[3])
#
#     # compute the eye aspect ratio
#     ear = (A + B) / (2.0 * C)
#
#     # return the eye aspect ratio
#     return ear
#
#
# # construct the argument parse and parse the arguments
# """
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
# 	help="path to facial landmark predictor")
# ap.add_argument("-v", "--video", type=str, default="",
# 	help="path to input video file")
# args = vars(ap.parse_args())
# """
#
# # shape=predictor(photo,detect[0])
#
# # define two constants, one for the eye aspect ratio to indicate
# # blink and then a second constant for the number of consecutive
# # frames the eye must be below the threshold
# EYE_AR_THRESH = 0.3
# EYE_AR_CONSEC_FRAMES = 3
#
# # initialize the frame counters and the total number of blinks
# COUNTER = 0
# TOTAL = 0
#
# # initialize dlib's face detector (HOG-based) and then create
# # the facial landmark predictor
# print("[INFO] loading facial landmark predictor...")
# detector = dlib.get_frontal_face_detector()
# # predictor = dlib.shape_predictor(args["shape_predictor"])
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# # grab the indexes of the facial landmarks for the left and
# # right eye, respectively
# (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
# (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
#
# # start the video stream thread
# print("[INFO] starting video stream thread...")
# # vs = FileVideoStream(args["video"]).start()
# # fileStream = True
# #vs = VideoStream(src=0).start()
# vs = cv2.VideoCapture(0)
# # vs = VideoStream(usePiCamera=True).start()
# fileStream = False
# time.sleep(1.0)
#
# # loop over frames from the video stream
# while True:
#     # if this is a file video stream, then we need to check if
#     # there any more frames left in the buffer to process
#     ret,frame = vs.read()
#
#     if fileStream and not ret:
#         break
#
#     # grab the frame from the threaded video file stream, resize
#     # it, and convert it to grayscale
#     # channels)
#
#     frame = imutils.resize(frame, width=450)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # detect faces in the grayscale frame
#     rects = detector(gray, 0)
#
#     # loop over the face detections
#     prev_y = 0
#     for rect in rects:
#         # determine the facial landmarks for the face region, then
#         # convert the facial landmark (x, y)-coordinates to a NumPy
#         # array
#         shape = predictor(gray, rect)
#         shape = face_utils.shape_to_np(shape)
#
#         # extract the left and right eye coordinates, then use the
#         # coordinates to compute the eye aspect ratio for both eyes
#         leftEye = shape[lStart:lEnd]
#         rightEye = shape[rStart:rEnd]
#         leftEAR = eye_aspect_ratio(leftEye)
#         rightEAR = eye_aspect_ratio(rightEye)
#
#         # average the eye aspect ratio together for both eyes
#         ear = (leftEAR + rightEAR) / 2.0
#
#         # compute the convex hull for the left and right eye, then
#         # visualize each of the eyes
#         leftEyeHull = cv2.convexHull(leftEye)
#         rightEyeHull = cv2.convexHull(rightEye)
#         cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
#         cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
#
#         # check to see if the eye aspect ratio is below the blink
#         # threshold, and if so, increment the blink frame counter
#         if ear < EYE_AR_THRESH:
#             COUNTER += 1
#             pyautogui.press('space')
#             # x, y, w, h = cv2.boundingRect(c)
#             # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             # if y < prev_y:
#             #     pyautogui.press('space')
#             #
#             # prev_y = y
#
#         # otherwise, the eye aspect ratio is not below the blink
#         # threshold
#         else:
#             # if the eyes were closed for a sufficient number of
#             # then increment the total number of blinks
#             if COUNTER >= EYE_AR_CONSEC_FRAMES:
#                 pyautogui.press('space')
#                 TOTAL += 1
#
#
#             # reset the eye frame counter
#             COUNTER = 0
#
#         # draw the total number of blinks on the frame along with
#         # the computed eye aspect ratio for the frame
#         cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#         cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#
#     # show the frame
#     cv2.imshow("Frame", frame)
#     key = cv2.waitKey(1) & 0xFF
#
#     # if the `q` key was pressed, break from the loop
#     if key == ord("q"):
#         break
#
# # do a bit of cleanup
# vs.release()
# cv2.destroyAllWindows()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#check mask
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import cv2
import pygame
import pyttsx3
from keras.models import load_model
speaker = pyttsx3.init()
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
threshold=0.90
cap=cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font=cv2.FONT_HERSHEY_COMPLEX
model = load_model('MyTrainingModel.h5')

def preprocessing(img):
    img=img.astype("uint8")
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img = img/255
    return img


def get_className(classNo):
	if classNo==0:
		return "Mask"
	elif classNo==1:


		return "No Mask"


while True:
	sucess, imgOrignal=cap.read()
	faces = facedetect.detectMultiScale(imgOrignal,1.3,5)
	for x,y,w,h in faces:
		# cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(50,50,255),2)
		# cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (50,50,255),-2)
		crop_img=imgOrignal[y:y+h,x:x+h]
		img=cv2.resize(crop_img, (32,32))
		img=preprocessing(img)
		img=img.reshape(1, 32, 32, 1)
		# cv2.putText(imgOrignal, "Class" , (20,35), font, 0.75, (0,0,255),2, cv2.LINE_AA)
		# cv2.putText(imgOrignal, "Probability" , (20,75), font, 0.75, (255,0,255),2, cv2.LINE_AA)
		prediction=model.predict(img)
		classIndex=model.predict_classes(img)
		probabilityValue=np.amax(prediction)
		if probabilityValue>threshold:
			if classIndex==0:
				cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,255,0),2)
				cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
				cv2.putText(imgOrignal, str(get_className(classIndex)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
			elif classIndex==1:
				pygame.init()
				pygame.mixer.music.load("alert.wav")
				pygame.mixer.music.play()
				cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(50,50,255),2)
				cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (50,50,255),-2)
				cv2.putText(imgOrignal, str(get_className(classIndex)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)

			# cv2.putText(imgOrignal,str(round(probabilityValue*100, 2))+"%" ,(180, 75), font, 0.75, (255,0,0),2, cv2.LINE_AA)
	cv2.imshow("Result",imgOrignal)
	k=cv2.waitKey(1)
	if k==ord('q'):
		break


cap.release()
cv2.destroyAllWindows()






