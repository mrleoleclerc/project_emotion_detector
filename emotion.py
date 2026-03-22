import cv2
from deepface import DeepFace

# каскадный классификатор лица (модель для поиска лиц)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# начало видео с вебкамеры (0)
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    # чб формат, чтобы искать лица быстрее
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # ргб формат для DeepFace
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
    # выделяем лица
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))
    for (x, y, w, h) in faces:
        # область лица
        face_rec = rgb_frame[y:y + h, x:x + w]
        # анализ лица
        result = DeepFace.analyze(face_rec, actions=['emotion'], enforce_detection=False)
        # выбираем наиболее предсказуемую эмоцию
        emotion = result[0]['dominant_emotion']
        # рамка с найденной эмоцией
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    # вывод заголовка
    cv2.imshow('emotion detector', frame)
    # q для выхода
    if cv2.waitKey(1) & 0xFF == ord('q'): # or emotion:
        print(emotion) # печатаем последнюю эмоцию
        break
# завершение работы
cap.release()
cv2.destroyAllWindows()