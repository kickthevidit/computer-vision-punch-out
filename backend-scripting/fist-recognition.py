import os
import sys
import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import constants
from typing import Union

def landmark_to_np_array(landmarks: "mp.framework.formats.landmark_pb2.NormalizedLandmarkList") -> Union[None, np.array]:
	"""
	Description: converts landmarks to a numpy array of dtype f16 and shape 21,3

	Finished currently
	"""

	# Create an array to store
	landmarks_array = [[landmark.x, landmark.y, landmark.z] for landmark in landmarks.landmark]
	landmarks_array = np.array(landmarks_array, dtype=np.dtype('f8')) # convert to np.array

	assert(landmarks_array.shape == (21,3))
	# print(landmarks_array)
	return landmarks_array

def refresh_feature_list(d_hand_l: bool, d_hand_r: bool, d_nose: bool) -> None:
	"""
	Description: printing whether nose, left hand and right hand are detected
	"""

	def detected(condition: bool) -> str:
		return "Detected" if condition else "Absent"

	os.system('clear')
	print(f"Frame {frame_no}")
	print(f"Detected features:\t|hand-l: {detected(d_hand_l)}|\t|hand-r: {detected(d_hand_r)}|\t| nose: {detected(d_nose)}|\n")

def check_model_args(y_n: bool) -> bool:
	"""
	Check command line argument signal to see if the right model should be run.
	"""

	return y_n != 0

if __name__ == "__main__":

	####  pre-processing, do not delete

	# checking command line arguments
	run_pose = None
	run_hands = None
	if len(sys.argv) > 1:
		first = 1
		for arg in sys.argv:
			if first:
				first = 0
				continue
			match arg:
				case "hands":
					print("arg hands")
					run_hands = 1
					run_pose = 0
				case "pose":
					print("arg pose")
					run_hands = 1
					run_pose = 1
				case "all": # Note: all does identify all landmarks
					print("arg all")
					run_hands = 1
					run_pose = 1
				case _:
					raise Exception(f"Inavlid Arg, {arg} is not a valid argument")
	
	
	os.system("clear")
	#### pre-processing, do not delete


	cap = cv2.VideoCapture(0)

	### landmarks to numpy array function test
	if check_model_args(run_hands): hands = mp.solutions.hands.Hands()
	if check_model_args(run_pose): pose = mp.solutions.pose.Pose(min_detection_confidence = 0.5)
	drawing_utils = mp.solutions.drawing_utils
	drawing_styles = mp.solutions.drawing_styles

	# only when pose
	l_hand_data: Union[None, list[tuple[int, Union[None, list[float, float]]]]] = []
	r_hand_data: Union[None, list[tuple[int, Union[None, list[float, float]]]]] = []
	nose_data: Union[None, list[tuple[int, Union[None, list[float, float]]]]] = []	

	# Run the model on a video stream.
	frame_no = 0
	# running loop for camera
	while True:
		# bools whether hands or nose are detected
		d_hand_l: bool = 0 if check_model_args(run_hands) else None
		d_hand_r: bool = 0 if check_model_args(run_hands) else None
		d_nose: bool = 0 if check_model_args(run_pose) else None
		frame_no += 1
	
		# Capture a frame from the video stream.
		ret, frame = cap.read()

		# Convert the frame to a NumPy array.
		frame = np.array(frame)

		print(frame.shape)

		# Run the model on the frame.
		if check_model_args(run_hands): results_hands = hands.process(frame)
		if check_model_args(run_pose): results_pose = pose.process(frame)
		# TODO: add python snippets pack
		# Analyze the output of the model to detect punches.
		if check_model_args(run_hands) and results_hands.multi_hand_landmarks:
			for hand_lr in results_hands.multi_handedness:
				if hand_lr.classification[0].label == "Left":
					d_hand_l = 1
				elif hand_lr.classification[0].label == "Right":
					d_hand_r = 1

			for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
				# draw hand_landmarks onto image
				drawing_utils.draw_landmarks(frame, hand_landmarks,mp.solutions.hands.HAND_CONNECTIONS)
		
		if check_model_args(run_pose) and results_pose.pose_landmarks:

			# TODO: does not work
			if results_pose.pose_landmarks.landmark[0].visibility > 0: d_nose = 1 

			# nose_data.append(frame, landmark_to_np_array(results_pose.pose_landmarks)[0][:2])

			drawing_utils.draw_landmarks(
				frame, results_pose.pose_landmarks,mp.solutions.pose.POSE_CONNECTIONS)

		# Terminal printing
		refresh_feature_list(d_hand_l, d_hand_r, d_nose)

		# Display the frame.
		cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
		cv2.resizeWindow('Video', constants.WINDOW_WIDTH, constants.WINDOW_HEIGHT)
		cv2.imshow('Video', frame)

		# Wait for a key press.
		key = cv2.waitKey(1)

		# If the user presses the `q` key, stop the program.
		if key == ord("q"):
			break
	
	# data_np_array = np.array(hand_data)
	# print(data_np_array.shape)
	# if (len(hand_data) != 0):
	# 	plt.plot(data_np_array[:,0,2])
		

	plt.show()

	# Release the video capture object.
	cap.release()

	# Close all windows.
	cv2.destroyAllWindows()