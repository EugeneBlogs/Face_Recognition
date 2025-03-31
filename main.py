# region Библиотеки
import datetime
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog, scrolledtext

import cv2
import numpy as np
from PIL import Image


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
            ("Добавить ученика (камера)", self.add_user),
            ("Добавить ученика (видео)", self.add_video),
            ("Запустить распознавание", self.start_recognition),
            ("Удалить ученика", self.delete_user),
            ("Обучить модель", self.training_model),
            ("История прохождений", self.show_history),
            ("Обновить список", self.update_users)
        ]

        for i, (text, command) in enumerate(buttons):
            btn = ModernButton(button_frame, text=text, command=command)
            btn.pack(fill=tk.X, pady=5, ipady=8)

        # Настройка веса строк и столбцов
        main_frame.rowconfigure(1, weight=1)
        main_frame.columnconfigure(0, weight=1)

        # Инициализация данных
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.users = []

        self.update_users()

    # endregion

    # region Работа с учениками
    def update_users(self):
        # Чистим данные и таблицу
        self.users = []
        for i in self.tree.get_children():
            self.tree.delete(i)

        try:
            with open("names.txt", "r+", encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if line:
                        i = line.split(",")
                        if len(i) >= 3:
                            user = [i[0], i[1], i[2]]
                            self.users.append(user)
                            self.tree.insert("", tk.END, values=user)
        except FileNotFoundError:
            messagebox.showwarning("Предупреждение", "Файл с данными учеников не найден.")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при чтении файла: {str(e)}")

    def get_images_and_labels(self, datapath):
        image_paths = [os.path.join(datapath, f) for f in os.listdir(datapath)]
        images = []
        labels = []

        for image_path in image_paths:
            try:
                image_pil = Image.open(image_path).convert('L')
                image = np.array(image_pil, 'uint8')
                nbr = int(os.path.split(image_path)[1].split(".")[0].replace("face-", ""))
                faces = self.faceCascade.detectMultiScale(image)

                for (x, y, w, h) in faces:
                    images.append(image[y: y + h, x: x + w])
                    labels.append(nbr)
                    cv2.imshow("Photo Analysis", image[y: y + h, x: x + w])
                    cv2.waitKey(100)
            except Exception as e:
                print(f"Ошибка обработки изображения {image_path}: {str(e)}")

        return images, labels

    def training_model(self):
        if len(self.tree.get_children()) != 0:
            try:
                images, labels = self.get_images_and_labels("dataSet")
                self.recognizer.train(images, np.array(labels))
                self.recognizer.save("trainer.yml")
                cv2.destroyAllWindows()
                messagebox.showinfo("Успешно", "Нейросеть закончила обучение.")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Произошла ошибка при обучении модели: {str(e)}")
        else:
            messagebox.showwarning("Предупреждение", "Нет данных для обучения.")

    def add_user(self):
        if not messagebox.askyesno("Подтверждение",
                                   "Процедура сканирования займёт некоторое время.\n"
                                   "В это время вам необходимо смотреть в камеру, не закрывать лицо и вращать головой.\n"
                                   "Продолжить?"):
            return

        name = simpledialog.askstring("Регистрация ученика", "Имя ученика:")
        if not name:
            return

        klass = simpledialog.askstring("Регистрация ученика", "Класс ученика (число+буква):")
        if not klass:
            return

        try:
            with open("names.txt", "r+", encoding='utf-8') as file:
                lines = file.readlines()
                last_line = lines[-1].strip() if lines else "0,-,-"
                last_id = int(last_line.split(",")[0])
                new_id = last_id + 1
                file.write(f"\n{new_id},{name},{klass}")

            detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
            offset = 50
            video = cv2.VideoCapture(0)

            for i in range(70):
                ret, im = video.read()
                if not ret:
                    continue

                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

                for (x, y, w, h) in faces:
                    try:
                        cv2.imwrite(f"dataSet/face-{new_id}.{i}.jpg",
                                    gray[y - offset:y + h + offset, x - offset:x + w + offset])
                        cv2.rectangle(im, (x - 50, y - 50), (x + w + 50, y + h + 50), (225, 0, 0), 2)
                        cv2.imshow('Creating Photo', im[y - offset:y + h + offset, x - offset:x + w + offset])
                    except Exception as e:
                        print(f"Ошибка сохранения фото: {str(e)}")

                    cv2.waitKey(100)

            video.release()
            cv2.destroyAllWindows()
            self.update_users()
            messagebox.showinfo("Успешно", f"Поздравляем, {name}!\nВы успешно зарегистрировались в системе.")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка: {str(e)}")
            try:
                with open("names.txt", "r+", encoding='utf-8') as file:
                    lines = file.readlines()
                    lines = [line for line in lines if line.strip() != f"{new_id},{name},{klass}"]
                    file.seek(0)
                    file.writelines(lines)
                    file.truncate()
            except:
                pass

    def add_video(self):
        name = simpledialog.askstring("Регистрация ученика", "Имя ученика:")
        if not name:
            return

        klass = simpledialog.askstring("Регистрация ученика", "Класс ученика (число+буква):")
        if not klass:
            return

        filename = filedialog.askopenfilename(
            title="Выберите видео",
            filetypes=(("Видео файлы", "*.mp4;*.mov;*.avi;*.mkv"), ("Все файлы", "*.*"))
        )

        if not filename:
            return

        try:
            with open("names.txt", "r+", encoding='utf-8') as file:
                lines = file.readlines()
                last_line = lines[-1].strip() if lines else "0,-,-"
                last_id = int(last_line.split(",")[0])
                new_id = last_id + 1
                file.write(f"\n{new_id},{name},{klass}")

            detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
            offset = 50
            video = cv2.VideoCapture(filename)
            i = 0

            while i < 70:
                ret, im = video.read()
                if not ret:
                    break

                try:
                    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

                    for (x, y, w, h) in faces:
                        cv2.imwrite(f"dataSet/face-{new_id}.{i}.jpg",
                                    gray[y - offset:y + h + offset, x - offset:x + w + offset])
                        cv2.rectangle(im, (x - 50, y - 50), (x + w + 50, y + h + 50), (225, 0, 0), 2)
                        cv2.imshow('Creating Photo', im[y - offset:y + h + offset, x - offset:x + w + offset])
                        i += 1
                        cv2.waitKey(1)

                except Exception as e:
                    print(f"Ошибка обработки кадра: {str(e)}")

            video.release()
            cv2.destroyAllWindows()

            if i >= 70:
                self.update_users()
                messagebox.showinfo("Успешно", f"Поздравляем, {name}!\nВы успешно зарегистрировались в системе.")
            else:
                self.cleanup_failed_registration(new_id, name, klass, i)
                messagebox.showerror("Ошибка",
                                     "В видео недостаточно кадров лица.\n"
                                     "Попробуйте перезаписать видео. Видео должно длиться от 15 секунд.\n"
                                     "Смотрите в камеру, не закрывайте лицо и вращайте головой.")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка: {str(e)}")

    def cleanup_failed_registration(self, user_id, name, klass, photos_taken):
        try:
            with open("names.txt", "r+", encoding='utf-8') as file:
                lines = file.readlines()
                lines = [line for line in lines if line.strip() != f"{user_id},{name},{klass}"]
                file.seek(0)
                file.writelines(lines)
                file.truncate()

            for i in range(photos_taken):
                try:
                    os.remove(f"dataSet/face-{user_id}.{i}.jpg")
                except:
                    pass

        except Exception as e:
            print(f"Ошибка очистки: {str(e)}")

    def delete_user(self):
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("Ошибка", "Выберите ученика в таблице.")
            return

        item = self.tree.item(selected_item[0])
        selected_people = item["values"]

        confirm = simpledialog.askstring(
            "Подтверждение удаления",
            f'Для удаления ученика введите его имя: "{selected_people[1]}"')

        if confirm != selected_people[1]:
            messagebox.showinfo("Отмена", "Удаление отменено.")
            return

        try:
            with open("names.txt", "r+", encoding='utf-8') as file:
                lines = file.readlines()
                lines = [line for line in lines if
                         line.strip() != f"{selected_people[0]},{selected_people[1]},{selected_people[2]}"]
                file.seek(0)
                file.writelines(lines)
                file.truncate()

            for i in range(70):
                try:
                    os.remove(f"dataSet/face-{selected_people[0]}.{i}.jpg")
                except:
                    pass

            self.update_users()
            messagebox.showinfo("Успешно", f"Ученик \"{selected_people[1]}\" удалён из базы.")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при удалении: {str(e)}")

    # endregion

    # region Работа с историей
    def show_history(self):
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

        save_btn = ModernButton(button_frame, text="Сохранить изменения", command=self.save_history)
        save_btn.pack(side=tk.RIGHT)

        clear_btn = ModernButton(button_frame, text="Очистить историю", command=self.clear_history)
        clear_btn.pack(side=tk.RIGHT, padx=5)

    def save_history(self):
        try:
            with open("history.txt", "w", encoding='utf-8') as file:
                file.write(self.history_text.get("1.0", tk.END))
            messagebox.showinfo("Успешно", "История обновлена.")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при сохранении: {str(e)}")

    def clear_history(self):
        if messagebox.askyesno("Подтверждение", "Вы уверены, что хотите очистить всю историю?"):
            self.history_text.delete("1.0", tk.END)

    # endregion

    # region Распознавание лиц
    def start_recognition(self):
        try:
            self.recognizer.read("trainer.yml")
            video = cv2.VideoCapture(0)
            font = cv2.FONT_HERSHEY_SIMPLEX

            while True:
                ret, im = video.read()
                if not ret:
                    continue

                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                faces = self.faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=5,
                    minSize=(100, 100),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

                for (x, y, w, h) in faces:
                    nbr_predicted, coord = self.recognizer.predict(gray[y:y + h, x:x + w])
                    student = None

                    for item in self.tree.get_children():
                        user = self.tree.item(item)["values"]
                        if int(nbr_predicted) == int(user[0]) and int(coord) < 70:
                            student = [user[1], user[2]]
                            break

                    current_time = datetime.datetime.now()
                    date = current_time.strftime("%d.%m.%Y %H:%M:%S")

                    if student:
                        color = (0, 255, 0)
                        log_entry = f"Ученик {student[1]} класса {student[0]} прошёл через проходную в {date}."
                    else:
                        color = (0, 0, 255)
                        log_entry = f"Неизвестный человек пытался пройти через проходную в {date}."

                    cv2.rectangle(im, (x, y), (x + w, y + h), color, 2)

                    try:
                        with open("history.txt", "a", encoding='utf-8') as file:
                            file.write(f"\n{log_entry}")
                    except:
                        pass

                    cv2.imshow('Face Recognizer ("Esc" to exit)', im)

                    key = cv2.waitKey(3000)
                    if key == 27:
                        video.release()
                        cv2.destroyAllWindows()
                        return

        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при распознавании: {str(e)}")
            try:
                video.release()
                cv2.destroyAllWindows()
            except:
                pass


# endregion

if __name__ == "__main__":
    app = App()
    app.mainloop()
