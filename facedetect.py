# coding:UTF-8
import cv2
import sys

if __name__ == '__main__':
    image_file = "test.jpg"
    cascade_file = "haarcascade_frontalface_alt.xml"
    
    # 入力画像読み込み
    image = cv2.imread(image_file)
    # グレースケールに変換
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 学習済みデータ読み込み
    cascade = cv2.CascadeClassifier(cascade_file)
    
    # 顔認識の実行
    face_list = cascade.detectMultiScale(image_gs,
                                         scaleFactor = 1.1,
                                         minNeighbors = 1,
                                         minSize = (28, 28))
    
    if len(face_list) > 0:
        print(face_list)
        color = (0, 0, 255)
        
        for face in face_list:
            x, y, w, h = face
            cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness=8)
        cv2.imwrite("result.png", image)
    else:
        print("no face")
