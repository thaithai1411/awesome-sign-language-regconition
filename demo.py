import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os


# Phát hiện thủ ngữ
def detect_gesture():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)

    while True:
        success, img = cap.read()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            hand_bbox = hand['bbox']
            fingers = hand['fingers']

            # Hiển thị hộp giới hạn cho bàn tay
            x, y, w, h = hand_bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

            # Hiển thị số ngón tay của bàn tay
            cv2.putText(img, str(len(fingers)), (x, y - 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        cv2.imshow("Image", img)

        key = cv2.waitKey(1)
        if key == 27:  # Bấm phím Esc để thoát
            break

    cap.release()
    cv2.destroyAllWindows()


# Chụp và lưu ảnh thủ ngữ "W"
def capture_gesture():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)

    offset = 20
    imgSize = 300

    target_folder = "datas/W"
    counter = 0

    while True:
        success, img = cap.read()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            imgCropShape = imgCrop.shape

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgReSize = cv2.resize(imgCrop, (wCal, imgSize))
                imgReSizeShape = imgReSize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgReSize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgReSize = cv2.resize(imgCrop, (imgSize, hCal))
                imgReSizeShape = imgReSize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgReSize

            # Hiển thị ảnh chụp được lên màn hình
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

            # Lưu ảnh khi nhấn phím "s"
            key = cv2.waitKey(1)
            if key == ord("s"):
                counter += 1
                img_path = os.path.join(target_folder, f'W_{time.time()}.jpg')
                cv2.imwrite(img_path, imgWhite)
                print(f"Đã lưu ảnh số {counter} vào {img_path}")

        cv2.imshow("Image", img)

        key = cv2.waitKey(1)
        if key == 27:  # Bấm phím Esc để thoát
            break

    cap.release()
    cv2.destroyAllWindows()


# Huấn luyện mô hình CNN
def train_model():
    import random
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation

    DIRECTORY = r"datas"
    CATEGORIES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                  'U', 'V', 'W', 'X', 'Y', 'Z']

    data = []

    for categories in CATEGORIES:
        folder = os.path.join(DIRECTORY, categories)
        label = CATEGORIES.index(categories)

        for img in os.listdir(folder):
            img = os.path.join(folder, img)
            img_arr = cv2.imread(img)
            img_arr = cv2.resize(img_arr, (224, 224))
            data.append([img_arr, label])

    random.shuffle(data)

    x = []
    y = []
    for features, label in data:
        x.append(features)
        y.append(label)

    x = np.array(x)
    y = np.array(y)
    x = x / 255

    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=x.shape[1:], activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(26, activation='softmax'))
    model.summary()

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_fit = model.fit(x, y, epochs=5, batch_size=128, validation_split=0.1)

    model.save('models/mymodel_18_7_17h43.h5')

    plt.plot(model_fit.history['accuracy'])
    plt.plot(model_fit.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(model_fit.history['loss'])
    plt.plot(model_fit.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


# Chạy các hàm cải tiến
if __name__ == "__main__":
    # Chạy phát hiện thủ ngữ
    detect_gesture()

    # Chạy chụp và lưu ảnh thủ ngữ "W"
    capture_gesture()

    # Chạy huấn luyện mô hình CNN
    train_model()
