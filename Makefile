make:
	@g++ facedetector.cpp -o facedetector -lopencv_core -lopencv_imgproc -lopencv_objdetect -lopencv_highgui -lopencv_features2d -lopencv_flann -lopencv_imgproc -lopencv_core

clean:
	@rm -f facedetector
