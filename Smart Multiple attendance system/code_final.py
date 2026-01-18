import cv2
import os
import sqlite3
from tkinter import *
from tkinter import messagebox
from PIL import Image
import face_recognition
import pickle
import datetime
import pandas as pd
from collections import defaultdict
import smtplib
from email.message import EmailMessage
from twilio.rest import Client

# Initialize DB (run this once or at program start)
def init_db():
    conn = sqlite3.connect('students2.db')
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS students (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        address TEXT,
        mobile TEXT NOT NULL
    )
    ''')
    conn.commit()
    conn.close()

init_db()

# --------- Functions ---------
def capture_images(student_id, name, address, mobile):
    if not student_id or not name or not mobile:
        messagebox.showerror("Input Error", "Please fill Student ID, Name, and Mobile fields.")
        return

    folder_path = f'dataset/{name}'
    os.makedirs(folder_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Capturing Faces - Press 's' to Save, 'q' to Quit", frame)
        print(f"[INFO] Capturing {count + 1}/20 images. Press 's' to save, 'q' to quit.")

        key = cv2.waitKey(1)
        if key == ord('s') and count < 20:
            img_path = os.path.join(folder_path, f"{name}{count+1}.jpg")
            cv2.imwrite(img_path, frame)
            count += 1
            print(f"Saved {img_path}")
        elif key == ord('q') or count == 20:
            break

    cap.release()
    cv2.destroyAllWindows()

    if count > 0:
        try:
            conn = sqlite3.connect('students2.db')
            c = conn.cursor()
            c.execute("INSERT INTO students2 (id, name, address, mobile) VALUES (?, ?, ?, ?)",
                      (student_id, name, address, mobile))
            conn.commit()
            conn.close()
            update_encodings()
            messagebox.showinfo("Success", f"{count} images captured for {name}.")
        except sqlite3.IntegrityError:
            messagebox.showerror("Error", "Student ID already exists!")
    else:
        messagebox.showwarning("Warning", "No images were captured.")

def update_encodings():
    known_encodings = []
    known_names = []
    for student_folder in os.listdir("dataset"):
        student_path = os.path.join("dataset", student_folder)
        if os.path.isdir(student_path):
            for image_file in os.listdir(student_path):
                img_path = os.path.join(student_path, image_file)
                img = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(img, model="cnn")
                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(student_folder)
    data = {"encodings": known_encodings, "names": known_names}
    with open("encodings.pickle", "wb") as f:
        pickle.dump(data, f)
    print("[INFO] Face encodings updated.")

def delete_student():
    student_id = entry_id.get().strip()
    if student_id:
        conn = sqlite3.connect('students.db')
        c = conn.cursor()
        c.execute("SELECT name FROM students WHERE id=?", (student_id,))
        student = c.fetchone()
        if student:
            name = student[0]
            folder_path = f'dataset/{name}'
            if os.path.exists(folder_path):
                for file in os.listdir(folder_path):
                    os.remove(os.path.join(folder_path, file))
                os.rmdir(folder_path)
            c.execute("DELETE FROM students WHERE id=?", (student_id,))
            conn.commit()
            conn.close()
            update_encodings()
            messagebox.showinfo("Deleted", f"Student with ID {student_id} deleted.")
        else:
            conn.close()
            messagebox.showwarning("Not Found", "Student ID not found.")
    else:
        messagebox.showwarning("Missing", "Please enter Student ID to delete.")

def take_attendance():
    try:
        with open("encodings.pickle", "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        messagebox.showerror("Error", "No face encodings found. Please register students first.")
        return

    conn = sqlite3.connect('students.db')
    c = conn.cursor()
    c.execute("SELECT id, name, address, mobile FROM students2")
    rows = c.fetchall()
    conn.close()

    student_info = {row[1]: {"id": row[0], "address": row[2], "mobile": row[3]} for row in rows}
    attendance = defaultdict(lambda: {"status": "Absent", "id": "", "address": "", "mobile": ""})
    for name, info in student_info.items():
        attendance[name] = {
            "status": "Absent",
            "id": info["id"],
            "address": info["address"],
            "mobile": info["mobile"]
        }

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    photo_folder = f"class_photos/{timestamp}"
    os.makedirs(photo_folder, exist_ok=True)

    print("[INFO] Starting webcam. Press 's' to save, 'q' to quit (Max 10 photos)...")
    cap = cv2.VideoCapture(0)
    captured = 0
    marked_present = set()

    while captured < 10:
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb_frame, model="cnn")
        encodings = face_recognition.face_encodings(rgb_frame, boxes)

        names = []
        for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Unknown"
            if True in matches:
                matchedIdxs = [i for i, b in enumerate(matches) if b]
                name_counts = {}
                for i in matchedIdxs:
                    matched_name = data["names"][i]
                    name_counts[matched_name] = name_counts.get(matched_name, 0) + 1
                name = max(name_counts, key=name_counts.get)
                names.append(name)
                attendance[name]["status"] = "Present"
                marked_present.add(name)

        for ((top, right, bottom, left), name) in zip(boxes, names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow("Classroom Attendance - Grid View", frame)
        print(f"[INFO] Capturing classroom photo {captured + 1}/10. Press 's' to save, 'q' to quit.")

        key = cv2.waitKey(1)
        if key == ord('s'):
            image_path = os.path.join(photo_folder, f"classroom_{captured+1}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"[INFO] Saved {image_path}")
            captured += 1
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"[INFO] Captured {captured} classroom images.")
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    time = datetime.datetime.now().strftime("%H:%M:%S")
    records = []
    for name, info in attendance.items():
        records.append({
            "ID": info["id"],
            "Name": name,
            "Address": info["address"],
            "Mobile": info["mobile"],
            "Date": date,
            "Time": time,
            "Status": info["status"]
        })
    df = pd.DataFrame(records)

    os.makedirs("attendance_reports", exist_ok=True)
    excel_path = f"attendance_reports/Attendance_{timestamp}.xlsx"
    df.to_excel(excel_path, index=False)
    print(f"[INFO] Attendance saved to: {excel_path}")
    send_email_report(excel_path)
    send_sms_to_absentees(excel_path)

def send_email_report(attachment_path):
    sender_email = 'Your email'
    sender_password = 'Your app password'  # Use app password or your actual password carefully
    receiver_email = 'Teacher email'
    subject = "Classroom Attendance Report"
    body = "Dear Teacher,\n\nPlease find attached the attendance report for today's class.\n\nRegards,\nSmart Attendance System"
    msg = EmailMessage()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.set_content(body)
    with open(attachment_path, 'rb') as f:
        file_data = f.read()
        file_name = os.path.basename(attachment_path)
    msg.add_attachment(file_data, maintype='application', subtype='octet-stream', filename=file_name)
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
        print("[INFO] Email sent successfully.")
    except Exception as e:
        print(f"[ERROR] Email failed: {e}")

def send_sms_to_absentees(attendance_file):
    account_sid = 'your_twilio_sid'
    auth_token = 'your_auth_token'
    twilio_number = 'your_twilio_phone_number'
    client = Client(account_sid, auth_token)
    df = pd.read_excel(attendance_file)
    absentees = df[df['Status'].str.lower() == 'absent']
    if absentees.empty:
        print("[INFO] No absentees found.")
        return
    for _, row in absentees.iterrows():
        try:
            msg = f"Alert: Your child {row['Name']} was absent today."
            message = client.messages.create(
                body=msg,
                from_=twilio_number,
                to='+91' + str(row['Mobile']) if not str(row['Mobile']).startswith('+') else str(row['Mobile'])

            )
            print(f"[INFO] SMS sent to {row['Name']} at {row['Mobile']}")
        except Exception as e:
            print(f"[ERROR] Failed SMS to {row['Name']} at {row['Mobile']}: {e}")

# --------- GUI ---------
root = Tk()
root.title("Smart Attendance System")
root.geometry("400x400")
root.config(bg="#f0f0f0")

Label(root, text="Student ID:", bg="#f0f0f0", font=("Arial", 12)).pack(pady=5)
entry_id = Entry(root, width=30, font=("Arial", 12))
entry_id.pack(pady=5)

Label(root, text="Student Name:", bg="#f0f0f0", font=("Arial", 12)).pack(pady=5)
entry_name = Entry(root, width=30, font=("Arial", 12))
entry_name.pack(pady=5)

Label(root, text="Address:", bg="#f0f0f0", font=("Arial", 12)).pack(pady=5)
entry_address = Entry(root, width=30, font=("Arial", 12))
entry_address.pack(pady=5)

Label(root, text="Mobile Number:", bg="#f0f0f0", font=("Arial", 12)).pack(pady=5)
entry_mobile = Entry(root, width=30, font=("Arial", 12))
entry_mobile.pack(pady=5)

Button(root, text="Register Student",
       command=lambda: capture_images(entry_id.get().strip(), entry_name.get().strip(), entry_address.get().strip(), entry_mobile.get().strip()),
       font=("Arial", 12), bg="#4CAF50", fg="white").pack(pady=10)

Button(root, text="Delete Student",
       command=delete_student,
       font=("Arial", 12), bg="#f44336", fg="white").pack(pady=5)

Button(root, text="Take Attendance",
       command=take_attendance,
       font=("Arial", 12), bg="#2196F3", fg="white").pack(pady=10)

root.mainloop()
