# region Библиотеки
import tkinter

import cv2  # Библиотека компьютерного зрения (необходимо установить пакет "opencv-python" и "opencv-contrib-python")
import os  # Библиотека для вызова системных функций
import numpy as np  # Библиотеку для обучения нейросетей

# Библиотеки для отображения интерфейса
from tkinter import *
from tkinter import ttk, filedialog
import customtkinter
from CustomTkinterMessagebox import *

from PIL import Image  # Библиотека для работы с изображениями (необходимо установить пакет "Pillow")

import datetime  # Библиотека для времени

# endregion

# region Базовые элементы

# Информация об окне
window = customtkinter.CTk()
window.title("Распознавание лиц")
window.geometry("800x550")

# Определяем базовые цвета интерфейса
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("green")

# Создаём рамку
frame = customtkinter.CTkFrame(master=window)
frame.pack(expand=True)

# Создаём информационные заголовки
lbl = customtkinter.CTkLabel(frame, text="   Распознавание лиц   ", font=("Calibri", 25))
lbl.grid(row=1, column=2, pady=5)

# endregion

# region Работа с учениками (таблицой)

users = []  # Массив учеников

tree = ttk.Treeview(frame, columns=("ID", "Name", "Class"), show="headings", height=20)  # Создаём таблицу
style = ttk.Style()
style.configure("Treeview.Heading", font=(None, 25))
style.configure("Treeview", font=(None, 15))
tree.grid(row=4, column=1)
# Определяем заголовки
tree.heading("ID", text="ID")
tree.heading("Name", text="Имя")
tree.heading("Class", text="Класс")


# Функция обновления учеников
def UpdateUsers():
    # Чистим данные и таблицу
    users = []
    for i in tree.get_children():
        tree.delete(i)
    file = open("names.txt", "r+", encoding='utf-8')  # Открываем файл с именами
    text = file.read()  # Считываем информацию с файла
    info = text.split("\n")  # Делим текст построчно
    try:
        for s in info:  # Проходим по всем строкам
            i = s.split(",")  # Создаём массив ID-Имя
            user = [i[0], i[1], i[2]]  # Создаём человека
            users.append(user)  # Добавляем его в общий массив
        for person in users:  # Добавляем людей в таблицу
            tree.insert("", END, values=person)
    except:
        print("Нет зарегистрировавшихся учеников.")


path = os.path.dirname(os.path.abspath(__file__))  # Получаем путь к скрипту
recognizer = cv2.face.LBPHFaceRecognizer_create()  # Создаём новый распознаватель лиц
faceCascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml")  # Указываем, что будем искать лица по примитивам Хаара


# Получаем фото и подписи из dataSet
def get_images_and_labels(datapath):
    image_paths = [os.path.join(datapath, f) for f in os.listdir(datapath)]  # Получаем путь к фото
    # Списки фото и подписей
    images = []
    labels = []
    # Перебираем все фото в dataSet
    for image_path in image_paths:
        image_pil = Image.open(image_path).convert('L')  # Читаем фото и переводим в ч/б
        image = np.array(image_pil, 'uint8')  # Переводим фото в numpy-массив
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("face-", ""))  # Получаем ID ученика из имени файла
        faces = faceCascade.detectMultiScale(image)  # Определяем лицо на фото
        for (x, y, w, h) in faces:  # Если лицо найдено, то
            images.append(image[y: y + h, x: x + w])  # Добавляем его к списку фото
            labels.append(nbr)  # Добавляем ID ученика в список подписей
            cv2.imshow("Photo Analysis", image[y: y + h, x: x + w])  # Выводим текущее фото на экран
            cv2.waitKey(100)  # Делаем паузу
    return images, labels  # Возвращаем список фото и подписей


# Функция для обучения модели
def TrainingModel():
    if len(tree.get_children()) != 0:
        images, labels = get_images_and_labels("dataSet")  # Получаем список фото и подписей
        recognizer.train(images, np.array(
            labels))  # Обучаем модель распознавания на наших фото и учим сопоставлять её лица и подписи к ним
        recognizer.save("trainer.yml")  # Сохраняем модель
        cv2.destroyAllWindows()  # Удаляем из памяти все созданные окна
        CTkMessagebox.messagebox(title="Успешно", text="Нейросеть закончила обучение.")


# Функция добавления нового ученика
def AddUser():
    CTkMessagebox.messagebox(title="Внимание!",
                             text='Процедура сканирования займёт некоторое время.\nВ это время вам необходимо смотреть в камеру, не закрывать лицо и вращать головой.\nНажмите "ОК", и через несколько секунд появится изображение с камеры.\nКак только оно появилось - начинайте движения головой.',
                             size="570x200")  # Информация для пользователя
    dialog = customtkinter.CTkInputDialog(text="Имя ученика:", title="Регистрация ученика")  # Запрашиваем имя человека
    name = dialog.get_input()
    if name == "" or name == None:  # Проверяем, чтобы поле не было пустым
        return 0
    dialog = customtkinter.CTkInputDialog(text="Класс ученика (число+буква):",
                                          title="Регистрация ученика")  # Запрашиваем класс ученика
    klass = dialog.get_input()
    if klass == "" or klass == None:  # Проверяем, чтобы поле не было пустым
        return 0

    path = os.path.dirname(os.path.abspath(__file__))  # Получаем путь к скрипту
    detector = cv2.CascadeClassifier(
        "haarcascade_frontalface_default.xml")  # Указываем, что будем искать лица по примитивам Хаара
    i = 0  # Счётчик изображений
    offset = 50  # Расстояния от распознанного лица до рамки

    file = open("names.txt", "r", encoding='utf-8')  # Открываем файл с именами для чтения
    text = file.read()  # Считываем информацию с файла
    file.close()  # Закрываем файл
    info = text.split("\n")  # Делим текст построчно
    if info[-1] != "":  # Если последняя строка не пустая (то есть люди есть), то
        last_str = info[-1].split(",")  # Делим последнюю строку по запятой
    else:  # Иначе
        last_str = [0, "-"]  # Имитируем последнюю строку с ID = 0
    ID = int(last_str[0]) + 1  # Новый ID - это старый ID + 1
    new = f"{ID},{name},{klass}"  # Новые данные
    text = f"{text}\n{new}"
    output = '\n'.join(line for line in text.split('\n') if line)  # Удаляем лишние пробелы
    file = open("names.txt", "w", encoding='utf-8')  # Открываем файл с именами для записи
    file.write(output)  # Добавляем новую информацию в файл
    file.close()  # Закрываем файл

    video = cv2.VideoCapture(0)  # Получаем доступ к камере

    # Запускаем цикл
    while True:
        ret, im = video.read()  # Получаем видеопоток
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # Переводим всё в ч/б для простоты
        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(
            100, 100))  # Настраиваем параметры распознавания и получаем лицо с камеры
        # Обрабатываем лица
        for (x, y, w, h) in faces:
            try:
                cv2.imwrite(f"dataSet/face-{ID}.{str(i)}.jpg",
                            gray[y - offset:y + h + offset, x - offset:x + w + offset])  # Записываем файл на диск
                cv2.rectangle(im, (x - 50, y - 50), (x + w + 50, y + h + 50), (225, 0, 0),
                              2)  # Формируем размеры окна для вывода лица
                cv2.imshow('Creating Photo', im[y - offset:y + h + offset,
                                             x - offset:x + w + offset])  # Показываем очередной кадр, который мы запомнили
                i += 1  # Увеличиваем счётчик кадров
                print(f"{i}/70")
            except:
                print(f"{i}/70")
            cv2.waitKey(100)  # Делаем паузу
        if i >= 70:  # Если у нас хватает кадров, то
            video.release()  # Освобождаем камеру
            cv2.destroyAllWindows()  # Удалаяем все созданные окна
            break  # Останавливаем цикл

    UpdateUsers()  # Обновляем таблицу
    CTkMessagebox.messagebox(title="Успешно", text=f"Поздравляем, {name}!\nВы успешно зарегистрировались в системе.")


# Функция добавления нового ученика с помощью видео
def AddVideo():
    dialog = customtkinter.CTkInputDialog(text="Имя ученика:", title="Регистрация ученика")  # Запрашиваем имя человека
    name = dialog.get_input()
    if name == "" or name == None:  # Проверяем, чтобы поле не было пустым
        return 0
    dialog = customtkinter.CTkInputDialog(text="Класс ученика (число+буква):",
                                          title="Регистрация ученика")  # Запрашиваем класс ученика
    klass = dialog.get_input()
    if klass == "" or klass == None:  # Проверяем, чтобы поле не было пустым
        return 0
    filename = filedialog.askopenfilename()
    words = filename.split('/')
    last_word = words[-1].split('.')
    format = last_word[-1].upper()
    formats = ["MP4", "MOV", "AVI", "MKV"]
    if format not in formats:
        txt_formats = ""
        for item in formats:
            txt_formats += f"{item}, "
        txt_formats = txt_formats[:-2]
        CTkMessagebox.messagebox(title="Ошибка!",
                                 text=f"Неподдерживаемый формат видео.\nВыберите файл формата: {txt_formats}.")
        return 0

    path = os.path.dirname(os.path.abspath(__file__))  # Получаем путь к скрипту
    detector = cv2.CascadeClassifier(
        "haarcascade_frontalface_default.xml")  # Указываем, что будем искать лица по примитивам Хаара
    i = 0  # Счётчик изображений
    offset = 50  # Расстояния от распознанного лица до рамки

    file = open("names.txt", "r", encoding='utf-8')  # Открываем файл с именами для чтения
    text = file.read()  # Считываем информацию с файла
    file.close()  # Закрываем файл
    info = text.split("\n")  # Делим текст построчно
    if info[-1] != "":  # Если последняя строка не пустая (то есть люди есть), то
        last_str = info[-1].split(",")  # Делим последнюю строку по запятой
    else:  # Иначе
        last_str = [0, "-"]  # Имитируем последнюю строку с ID = 0
    ID = int(last_str[0]) + 1  # Новый ID - это старый ID + 1
    new = f"{ID},{name},{klass}"  # Новые данные
    text = f"{text}\n{new}"
    output = '\n'.join(line for line in text.split('\n') if line)  # Удаляем лишние пробелы
    file = open("names.txt", "w", encoding='utf-8')  # Открываем файл с именами для записи
    file.write(output)  # Добавляем новую информацию в файл
    file.close()  # Закрываем файл

    video = cv2.VideoCapture(filename)  # Получаем доступ к видео
    count_fps = 0

    # Запускаем цикл
    while True:
        ret, im = video.read()  # Получаем видеопоток
        count_fps += 1
        if count_fps == 11:
            count_fps = 1
        try:
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # Переводим всё в ч/б для простоты
            faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(
                100, 100))  # Настраиваем параметры распознавания и получаем лицо с камеры
            # Обрабатываем лица
            for (x, y, w, h) in faces:
                try:
                    if count_fps == 1:
                        cv2.imwrite(f"dataSet/face-{ID}.{str(i)}.jpg",
                                    gray[y - offset:y + h + offset,
                                    x - offset:x + w + offset])  # Записываем файл на диск
                        cv2.rectangle(im, (x - 50, y - 50), (x + w + 50, y + h + 50), (225, 0, 0),
                                      2)  # Формируем размеры окна для вывода лица
                        cv2.imshow('Creating Photo', im[y - offset:y + h + offset,
                                                     x - offset:x + w + offset])  # Показываем очередной кадр, который мы запомнили
                        i += 1  # Увеличиваем счётчик кадров
                        print(f"{i}/70")
                except:
                    print(f"{i}/70")
                cv2.waitKey(1)  # Делаем паузу
            if i >= 70:
                video.release()  # Освобождаем камеру
                cv2.destroyAllWindows()  # Удалаяем все созданные окна
                UpdateUsers()  # Обновляем таблицу
                CTkMessagebox.messagebox(title="Успешно",
                                         text=f"Поздравляем, {name}!\nВы успешно зарегистрировались в системе.")
                break  # Останавливаем цикл
        except:
            # Удаляем ученика из файла
            file = open("names.txt", "r", encoding='utf-8')  # Открываем файл с именами для чтения
            text = file.read()  # Считываем информацию с файла
            file.close()  # Закрываем файл
            text = text.replace(f"{ID},{name},{klass}",
                                "")  # Удаляем ID и имя человека
            output = '\n'.join(line for line in text.split('\n') if line)  # Удаляем лишние пробелы
            file = open("names.txt", "w", encoding='utf-8')  # Открываем файл с именами для записи
            file.write(output)  # Добавляем новую информацию в файл
            file.close()  # Закрываем файл

            # Удаляем то количество фотографий, которые успели сделать
            for i in range(0, i):
                try:
                    os.remove(f"dataSet/face-{ID}.{str(i)}.jpg")
                except:
                    print("Файл уже удалён.")

            CTkMessagebox.messagebox(title="Ошибка!",
                                     text=f"В видео недостаточно кадров лица.\nПопробуйте перезаписать видео. Видео должно длится от 15 секунд.\nСмотрите в камеру, не закрывайте лицо и вращайте головой.",
                                     size="520x200")

            video.release()  # Освобождаем камеру
            cv2.destroyAllWindows()  # Удалаяем все созданные окна
            break  # Останавливаем цикл


# Функция удаления ученика
def DeleteUser():
    # Определяем, какого ученика выбрали
    selected_people = []
    for selected_item in tree.selection():
        item = tree.item(selected_item)
        selected_people = item["values"]
    # Проверяем, выбрана ли строка
    try:
        ID = int(selected_people[0])
        # Удаляем ученика после дополнительного вопроса
        dialog = customtkinter.CTkInputDialog(text=f'Для удаления ученика напишите его имя: "{selected_people[1]}".',
                                              title="Удаление ученика")
        name = dialog.get_input()
        if name == selected_people[1]:
            # Удаляем ученика из файла
            file = open("names.txt", "r", encoding='utf-8')  # Открываем файл с именами для чтения
            text = file.read()  # Считываем информацию с файла
            file.close()  # Закрываем файл
            text = text.replace(f"{selected_people[0]},{selected_people[1]},{selected_people[2]}",
                                "")  # Удаляем ID и имя человека
            output = '\n'.join(line for line in text.split('\n') if line)  # Удаляем лишние пробелы
            file = open("names.txt", "w", encoding='utf-8')  # Открываем файл с именами для записи
            file.write(output)  # Добавляем новую информацию в файл
            file.close()  # Закрываем файл
            UpdateUsers()  # Обновляем таблицу

            # Удаляем 70 фотографий ученика из базы
            for i in range(0, 70):
                os.remove(f"dataSet/face-{selected_people[0]}.{str(i)}.jpg")

            CTkMessagebox.messagebox(title="Успешно",
                                     text=f"Ученик \"{selected_people[1]}\" удалён из базы.")  # Выводим результат
    except:
        CTkMessagebox.messagebox(title="Ошибка!",
                                 text="Выберите ученика в таблице.")


# endregion

# region Работа с историей

# Показ истории
def ShowHistory():
    global entry
    # Создаём дополнительное окно
    root = customtkinter.CTkToplevel()
    root.title("История прохождений")
    root.geometry("800x550")
    entry = customtkinter.CTkTextbox(root, width=800, height=500, corner_radius=30, text_color="white")
    # Определяем функции для работы с текстом
    entry.bind('<Control-C>', copy_text)
    entry.bind('<Control-V>', paste_text)
    entry.bind('<Control-A>', select_text)
    entry.pack()
    file = open("history.txt", "r+", encoding='utf-8')  # Открываем файл с историей
    text = file.read()  # Считываем информацию с файла
    entry.insert(tkinter.END, text)  # Добавляем текст в поле
    save_btn = customtkinter.CTkButton(root, text="Сохранить изменения", font=("Calibri", 14), cursor="hand2",
                                       command=SaveHistory)
    save_btn.pack(pady=5)


# Функции для работы с текстом
def copy_text():
    text = entry.selection_get()
    window.clipboard_clear()
    window.clipboard_append(text)


def paste_text():
    text = window.clipboard_get()
    entry.insert(tkinter.INSERT, text)


def select_text():
    entry.tag_add(SEL, "1.0", END)
    entry.mark_set(INSERT, "1.0")
    entry.see(INSERT)


# Сохранение истории
def SaveHistory():
    file = open("history.txt", "w", encoding='utf-8')  # Открываем файл с историей для записи
    file.write(entry.get(1.0, 'end'))  # Добавляем новую информацию в файл
    file.close()  # Закрываем файл
    CTkMessagebox.messagebox(title="Успешно", text="История обновлена.")


# endregion

# region Запуск системы
def StartRecognition():
    path = os.path.dirname(os.path.abspath(__file__))  # Получаем путь к скрипту
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # Создаём новый распознаватель лиц
    recognizer.read("trainer.yml")  # Добавляем в него модель, которую мы обучили
    faceCascade = cv2.CascadeClassifier(
        "haarcascade_frontalface_default.xml")  # Указываем, что мы будем искать лица по примитивам Хаара

    video = cv2.VideoCapture(0)  # Получаем доступ к камере

    loop = True
    # Запускаем цикл
    while loop == True:
        ret, im = video.read()  # Получаем видеопоток
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # Переводим его в ч/б
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100),
                                             flags=cv2.CASCADE_SCALE_IMAGE)  # Определяем лица на видео
        # Перебираем все найденные лица
        for (x, y, w, h) in faces:
            nbr_predicted, coord = recognizer.predict(gray[y:y + h, x:x + w])  # Получаем ID ученика
            student = []  # Переменная для студента
            for item in tree.get_children():
                user = tree.item(item)["values"]
                if int(nbr_predicted) == int(user[0]) and int(coord) < 70:
                    student = [user[1], user[2]]
            # Получаем текущее время
            current_time = datetime.datetime.now()
            date = f"{current_time.day}.{current_time.month}.{current_time.year} {current_time.hour}:{current_time.minute}:{current_time.second}"
            # Если мы знаем имя ученика, то
            try:
                name = student[0]
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0),
                              2)  # Рисуем прямоугольник вокруг лица зелёного цвета
                file = open("history.txt", "r+", encoding='utf-8')  # Открываем файл с записями проходов для чтения
                # Считываем информацию
                now = file.read()
                file.close()  # Закрываем файл
                new = f"Ученик {student[1]} класса {student[0]} прошёл через проходную в {date}."
                text = f"{now}\n{new}"
                file = open("history.txt", "w", encoding='utf-8')  # Открываем файл с записями проходов для записи
                # Добавляем новую информацию в файл
                file.write(text)
                file.close()  # Закрываем файл
            except:
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255),
                              2)  # Рисуем прямоугольник вокруг лица красного цвета
                file = open("history.txt", "r+", encoding='utf-8')  # Открываем файл с записями проходов для чтения
                # Считываем информацию
                now = file.read()
                file.close()  # Закрываем файл
                new = f"Неизвестный человек пытался пройти через проходную в {date}."
                text = f"{now}\n{new}"
                file = open("history.txt", "w", encoding='utf-8')  # Открываем файл с записями проходов для записи
                # Добавляем новую информацию в файл
                file.write(text)
                file.close()  # Закрываем файл
            cv2.imshow('Face Recognizer ("Esc" to exit)', im)  # Выводим окно с изображением с камеры
            k = cv2.waitKey(3000)  # Делаем паузу в 3 секунды
            if k == 27:
                loop = False
                cv2.destroyAllWindows()
                break


# endregion

# region Кнопки
add_btn = customtkinter.CTkButton(frame, text="Добавить ученика с помощью камеры", font=("Calibri", 14), cursor="hand2",
                                  command=AddUser)
add_btn.grid(row=2, column=3, pady=5)

add_btn = customtkinter.CTkButton(frame, text="Добавить ученика с помощью видео", font=("Calibri", 14), cursor="hand2",
                                  command=AddVideo)
add_btn.grid(row=3, column=3, pady=5)

open_btn = customtkinter.CTkButton(frame, text="Запустить распознавание лиц", font=("Calibri", 14), cursor="hand2",
                                   command=StartRecognition)
open_btn.grid(row=4, column=3, pady=5)

del_btn = customtkinter.CTkButton(frame, text="Удалить ученика", font=("Calibri", 14), cursor="hand2",
                                  command=DeleteUser)
del_btn.grid(row=5, column=3, pady=5)

go_btn = customtkinter.CTkButton(frame, text="Обновить данные", font=("Calibri", 14), cursor="hand2",
                                 command=TrainingModel)
go_btn.grid(row=6, column=3, pady=5)

history_btn = customtkinter.CTkButton(frame, text="История прохождений", font=("Calibri", 14),
                                      cursor="hand2",
                                      command=ShowHistory)
history_btn.grid(row=7, column=3, pady=5)

# endregion

# region Запуск программы

UpdateUsers()  # Запускаем метод, чтобы обновить людей
window.mainloop()  # Запускаем бесконечный цикл окна

# endregion
