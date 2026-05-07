import cv2
import math

faceProto = "models/opencv_face_detector.pbtxt"
faceModel = "models/opencv_face_detector_uint8.pb"

ageProto = "models/age_deploy.prototxt"
ageModel = "models/age_net.caffemodel"

genderProto = "models/gender_deploy.prototxt"
genderModel = "models/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

video = cv2.VideoCapture(0)

padding = 20

while True:
    hasFrame, frame = video.read()

    if not hasFrame:
        break

    frameCopy = frame.copy()
    frameHeight = frameCopy.shape[0]
    frameWidth = frameCopy.shape[1]

    blob = cv2.dnn.blobFromImage(
        frameCopy,
        1.0,
        (300, 300),
        [104, 117, 123],
        True,
        False
    )

    faceNet.setInput(blob)
    detections = faceNet.forward()

    faceBoxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.7:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)

            faceBoxes.append([x1, y1, x2, y2])

            cv2.rectangle(frameCopy, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for faceBox in faceBoxes:
        face = frame[
            max(0, faceBox[1] - padding):
            min(faceBox[3] + padding, frameHeight - 1),

            max(0, faceBox[0] - padding):
            min(faceBox[2] + padding, frameWidth - 1)
        ]

        blob = cv2.dnn.blobFromImage(
            face,
            1.0,
            (227, 227),
            MODEL_MEAN_VALUES,
            swapRB=False
        )

        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        label = f"{gender}, {age}"

        cv2.putText(
            frameCopy,
            label,
            (faceBox[0], faceBox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

    cv2.imshow("Age Gender Detection", frameCopy)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()