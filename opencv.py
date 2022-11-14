import cv2

eye_img = "images/eye_img.png"
face_img = "images/face_img.png"


cascade_file = "haarcascade_eye.xml"
cascade = cv2.CascadeClassifier(cascade_file)

face_img = cv2.imread(face_img)
eye_img = cv2.imread(eye_img)

face_list = cascade.detectMultiScale(face_img)


def make(n):
    x = face_list[n][0]
    y = face_list[n][1]
    w = face_list[n][2]
    h = face_list[n][3]

    resized_eye_img = cv2.resize(eye_img, dsize=(w, h))
    cv2.imwrite("images/resize_eye.png", resized_eye_img)
    face_img[y : h + y, x : w + x] = cv2.imread("images/resize_eye.png")


make(0)
make(1)
cv2.imwrite("images/face_eye.png", face_img)
