# region Библиотеки
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog, scrolledtext
import cv2
import os
import numpy as np
from PIL import Image
import datetime


# endregion

# region Базовые элементы
class ModernButton(tk.Button):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.config(
            bg="#4CAF50",
            fg="white",
            activebackground="#45a049",
            activeforeground="white",
            relief=tk.FLAT,
            font=("Segoe UI", 10),
            padx=10,
            pady=5,
            cursor="hand2"
        )
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, e):
        self.config(bg="#45a049")

    def on_leave(self, e):
        self.config(bg="#4CAF50")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Распознавание лиц")
        self.geometry("900x600")
        self.configure(bg="#f0f0f0")

        # Стилизация
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Настройка стилей
        self.style.configure("Treeview.Heading", font=("Segoe UI", 12, "bold"), background="#4CAF50",
                             foreground="white")
        self.style.configure("Treeview", font=("Segoe UI", 11), rowheight=25, fieldbackground="#f0f0f0")
        self.style.map("Treeview", background=[("selected", "#4CAF50")])

        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TLabel", background="#f0f0f0", font=("Segoe UI", 11))

        self.create_widgets()

        # Инициализация данных
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.users = []

        self.UpdateUsers()

    def create_widgets(self):
        # Основной фрейм
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Заголовок
        header = ttk.Label(main_frame, text="Распознавание лиц", font=("Segoe UI", 20, "bold"))
        header.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Таблица учеников
        self.tree = ttk.Treeview(main_frame, columns=("ID", "Name", "Class"), show="headings", height=20)
        self.tree.grid(row=1, column=0, padx=(0, 20), sticky="nsew")

        # Настройка заголовков таблицы
        self.tree.heading("ID", text="ID")
        self.tree.heading("Name", text="Имя")
        self.tree.heading("Class", text="Класс")

        # Настройка столбцов таблицы
        self.tree.column("ID", width=50, anchor=tk.CENTER)
        self.tree.column("Name", width=200, anchor=tk.W)
        self.tree.column("Class", width=100, anchor=tk.CENTER)

        # Фрейм для кнопок
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=1, sticky="n")

        # Кнопки
        buttons = [
            ("Добавить ученика (камера)", self.AddUser),
            ("Добавить ученика (видео)", self.AddVideo),
            ("Запустить распознавание", self.StartRecognition),
            ("Удалить ученика", self.DeleteUser),
            ("Обучить модель", self.TrainingModel),
            ("История прохождений", self.ShowHistory)
        ]

        for i, (text, command) in enumerate(buttons):
            btn = ModernButton(button_frame, text=text, command=command)
            btn.pack(fill=tk.X, pady=5, ipady=8)

        # Настройка веса строк и столбцов
        main_frame.rowconfigure(1, weight=1)
        main_frame.columnconfigure(0, weight=1)

    # endregion

    # region Работа с учениками (таблицой)
    def UpdateUsers(self):
        # Чистим данные и таблицу
        self.users = []
        for i in self.tree.get_children():
            self.tree.delete(i)
        file = open("names.txt", "r+", encoding='utf-8')  # Открываем файл с именами
        text = file.read()  # Считываем информацию с файла
        info = text.split("\n")  # Делим текст построчно
        try:
            for s in info:  # Проходим по всем строкам
                i = s.split(",")  # Создаём массив ID-Имя
                user = [i[0], i[1], i[2]]  # Создаём человека
                self.users.append(user)  # Добавляем его в общий массив
            for person in self.users:  # Добавляем людей в таблицу
                self.tree.insert("", tk.END, values=person)
        except:
            print("Нет зарегистрировавшихся учеников.")

    def get_images_and_labels(self, datapath):
        image_paths = [os.path.join(datapath, f) for f in os.listdir(datapath)]  # Получаем путь к фото
        # Списки фото и подписей
        images = []
        labels = []
        # Перебираем все фото в dataSet
        for image_path in image_paths:
            image_pil = Image.open(image_path).convert('L')  # Читаем фото и переводим в ч/б
            image = np.array(image_pil, 'uint8')  # Переводим фото в numpy-массив
            nbr = int(
                os.path.split(image_path)[1].split(".")[0].replace("face-", ""))  # Получаем ID ученика из имени файла
            faces = self.faceCascade.detectMultiScale(image)  # Определяем лицо на фото
            for (x, y, w, h) in faces:  # Если лицо найдено, то
                images.append(image[y: y + h, x: x + w])  # Добавляем его к списку фото
                labels.append(nbr)  # Добавляем ID ученика в список подписей
                cv2.imshow("Photo Analysis", image[y: y + h, x: x + w])  # Выводим текущее фото на экран
                cv2.waitKey(100)  # Делаем паузу
        return images, labels  # Возвращаем список фото и подписей

    def TrainingModel(self):
        if len(self.tree.get_children()) != 0:
            images, labels = self.get_images_and_labels("dataSet")  # Получаем список фото и подписей
            self.recognizer.train(images, np.array(labels))  # Обучаем модель распознавания
            self.recognizer.save("trainer.yml")  # Сохраняем модель
            cv2.destroyAllWindows()  # Удаляем из памяти все созданные окна
            messagebox.showinfo("Успешно", "Нейросеть закончила обучение.")

    def AddUser(self):
        messagebox.showinfo("Внимание!",
                            'Процедура сканирования займёт некоторое время.\nВ это время вам необходимо смотреть в камеру, не закрывать лицо и вращать головой.\nНажмите "ОК", и через несколько секунд появится изображение с камеры.\nКак только оно появилось - начинайте движения головой.')  # Информация для пользователя

        name = simpledialog.askstring("Регистрация ученика", "Имя ученика:")  # Запрашиваем имя человека
        if not name:  # Проверяем, чтобы поле не было пустым
            return

        klass = simpledialog.askstring("Регистрация ученика",
                                       "Класс ученика (число+буква):")  # Запрашиваем класс ученика
        if not klass:  # Проверяем, чтобы поле не было пустым
            return

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

        self.UpdateUsers()  # Обновляем таблицу
        messagebox.showinfo("Успешно", f"Поздравляем, {name}!\nВы успешно зарегистрировались в системе.")

    def AddVideo(self):
        name = simpledialog.askstring("Регистрация ученика", "Имя ученика:")  # Запрашиваем имя человека
        if not name:  # Проверяем, чтобы поле не было пустым
            return

        klass = simpledialog.askstring("Регистрация ученика",
                                       "Класс ученика (число+буква):")  # Запрашиваем класс ученика
        if not klass:  # Проверяем, чтобы поле не было пустым
            return

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
            messagebox.showerror("Ошибка!",
                                 f"Неподдерживаемый формат видео.\nВыберите файл формата: {txt_formats}.")
            return

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
                    self.UpdateUsers()  # Обновляем таблицу
                    messagebox.showinfo("Успешно",
                                        f"Поздравляем, {name}!\nВы успешно зарегистрировались в системе.")
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

                messagebox.showerror("Ошибка!",
                                     f"В видео недостаточно кадров лица.\nПопробуйте перезаписать видео. Видео должно длится от 15 секунд.\nСмотрите в камеру, не закрывайте лицо и вращайте головой.",
                                     )

                video.release()  # Освобождаем камеру
                cv2.destroyAllWindows()  # Удалаяем все созданные окна
                break  # Останавливаем цикл

    def DeleteUser(self):
        # Определяем, какого ученика выбрали
        selected_people = []
        for selected_item in self.tree.selection():
            item = self.tree.item(selected_item)
            selected_people = item["values"]
        # Проверяем, выбрана ли строка
        try:
            ID = int(selected_people[0])
            # Удаляем ученика после дополнительного вопроса
            name = simpledialog.askstring("Удаление ученика",
                                          f'Для удаления ученика напишите его имя: "{selected_people[1]}".')
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
                self.UpdateUsers()  # Обновляем таблицу

                # Удаляем 70 фотографий ученика из базы
                for i in range(0, 70):
                    os.remove(f"dataSet/face-{selected_people[0]}.{str(i)}.jpg")

                messagebox.showinfo("Успешно",
                                    f"Ученик \"{selected_people[1]}\" удалён из базы.")  # Выводим результат
        except:
            messagebox.showerror("Ошибка!",
                                 "Выберите ученика в таблице.")

    # endregion

    # region Работа с историей
    def ShowHistory(self):
        history_window = tk.Toplevel(self)
        history_window.title("История прохождений")
        history_window.geometry("800x600")

        text_frame = ttk.Frame(history_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.history_text = scrolledtext.ScrolledText(
            text_frame,
            wrap=tk.WORD,
            font=("Segoe UI", 11),
            padx=10,
            pady=10
        )
        self.history_text.pack(fill=tk.BOTH, expand=True)

        try:
            with open("history.txt", "r", encoding='utf-8') as file:
                self.history_text.insert(tk.END, file.read())
        except FileNotFoundError:
            self.history_text.insert(tk.END, "Файл истории не найден.")

        button_frame = ttk.Frame(history_window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        save_btn = ModernButton(button_frame, text="Сохранить изменения", command=self.SaveHistory)
        save_btn.pack(side=tk.RIGHT)

    def SaveHistory(self):
        file = open("history.txt", "w", encoding='utf-8')  # Открываем файл с историей для записи
        file.write(self.history_text.get("1.0", tk.END))  # Добавляем новую информацию в файл
        file.close()  # Закрываем файл
        messagebox.showinfo("Успешно", "История обновлена.")

    # endregion

    # region Запуск системы
    def StartRecognition(self):
        path = os.path.dirname(os.path.abspath(__file__))  # Получаем путь к скрипту
        self.recognizer.read("trainer.yml")  # Добавляем в него модель, которую мы обучили

        video = cv2.VideoCapture(0)  # Получаем доступ к камере

        loop = True
        # Запускаем цикл
        while loop == True:
            ret, im = video.read()  # Получаем видеопоток
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # Переводим его в ч/б
            faces = self.faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100),
                                                      flags=cv2.CASCADE_SCALE_IMAGE)  # Определяем лица на видео
            # Перебираем все найденные лица
            for (x, y, w, h) in faces:
                nbr_predicted, coord = self.recognizer.predict(gray[y:y + h, x:x + w])  # Получаем ID ученика
                student = []  # Переменная для студента
                for item in self.tree.get_children():
                    user = self.tree.item(item)["values"]
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

if __name__ == "__main__":
    app = App()
    app.mainloop()
