import streamlit as st
import cv2
import face_recognition
import numpy as np
import os
import pandas as pd
from datetime import datetime
import mediapipe as mp
import time

# --------- Constants ---------
STUDENT_CREDENTIALS_FILE = "student_details/student_credentials.csv"
STUDENT_DETAILS_FILE = "student_details/student_details.csv"
ADMIN_CREDENTIALS_FILE = "student_details/admin_credentials.csv"
ATTENDANCE_FILE = "attendance1.xlsx"

# --------- Directories Setup ---------
os.makedirs("faces", exist_ok=True)
os.makedirs("student_details", exist_ok=True)

# --------- Session State ---------
for key in ["admin_logged_in", "student_logged_in", "student_user", "change_password", "registering_student", "show_login"]:
    if key not in st.session_state:
        st.session_state[key] = False if key not in ["student_user", "registering_student", "show_login"] else None

# --------- Initialize Admin ---------
def initialize_admin_credentials():
    if not os.path.exists(ADMIN_CREDENTIALS_FILE):
        admin_df = pd.DataFrame([["admin", "admin123"]], columns=["Username", "Password"])
        admin_df.to_csv(ADMIN_CREDENTIALS_FILE, index=False)

initialize_admin_credentials()

# --------- Helpers ---------
def safe_write_csv(df, file_path, mode="a", header=True):
    try:
        df.to_csv(file_path, mode=mode, header=header, index=False)
    except PermissionError:
        st.error(f"Permission denied while writing to {file_path}. Please close it if it's open in another program (e.g., Excel) and try again.")
        raise

# --------- Face and Attendance Functions ---------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
MOVEMENT_THRESHOLD = 2.0
MOVEMENT_FRAMES_REQUIRED = 10

def save_student_details(name, student_id, frame, password):
    file_path = f"faces/{student_id}_{name}.jpg"
    cv2.imwrite(file_path, frame)

    details_df = pd.DataFrame([[name, student_id]], columns=["Name", "Enrollment Number"])
    creds_df = pd.DataFrame([[student_id, password]], columns=["Enrollment Number", "Password"])

    safe_write_csv(details_df, STUDENT_DETAILS_FILE, mode="a", header=not os.path.exists(STUDENT_DETAILS_FILE))
    safe_write_csv(creds_df, STUDENT_CREDENTIALS_FILE, mode="a", header=not os.path.exists(STUDENT_CREDENTIALS_FILE))

def register_student():
    st.subheader("Register New Student")
    name = st.text_input("Student Name")
    student_id = st.text_input("Student ID")
    password = st.text_input("Set Password", type="password")

    if st.button("Capture Student Image"):
        if name and student_id and password:
            if os.path.exists(STUDENT_DETAILS_FILE):
                student_data = pd.read_csv(STUDENT_DETAILS_FILE)
                if student_id in student_data["Enrollment Number"].astype(str).values:
                    st.error("Student ID already exists.")
                    return

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Failed to access the camera.")
                return
            
            st.info("Press 'S' to capture the image.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame.")
                    break

                cv2.imshow("Capture Student Image - Press 'S' to Capture", frame)
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    save_student_details(name, student_id, frame, password)
                    st.success(f"Student {name} registered successfully!")
                    st.session_state.registering_student = False
                    break

            cap.release()
            cv2.destroyAllWindows()
        else:
            st.error("All fields are required!")

def load_registered_faces():
    encodings, names, ids = [], [], []
    for file in os.listdir("faces"):
        if file.endswith(".jpg"):
            parts = file.split("_")
            if len(parts) < 2:
                continue
            student_id = parts[0]
            name = "_".join(parts[1:]).split(".")[0]
            image = face_recognition.load_image_file(os.path.join("faces", file))
            encoding = face_recognition.face_encodings(image)
            if encoding:
                encodings.append(encoding[0])
                names.append(name)
                ids.append(student_id)
    return encodings, names, ids

def mark_attendance(name, student_id):
    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_excel(ATTENDANCE_FILE)
    else:
        df = pd.DataFrame(columns=["Student Name", "Student ID"])

    if student_id not in df["Student ID"].astype(str).values:
        new_row = pd.DataFrame([[name, student_id]], columns=["Student Name", "Student ID"])
        df = pd.concat([df, new_row], ignore_index=True)

    if today not in df.columns:
        df[today] = ""

    row_idx = df[df["Student ID"].astype(str) == str(student_id)].index[0]
    if pd.isna(df.loc[row_idx, today]) or df.loc[row_idx, today] == "":
        df.loc[row_idx, today] = current_time
        df.to_excel(ATTENDANCE_FILE, index=False)
        st.success(f"Attendance marked for {name}")
    else:
        st.info("Attendance already marked today.")

def get_nose_tip_position(landmarks, frame_shape):
    idx = 1
    x = int(landmarks[idx].x * frame_shape[1])
    y = int(landmarks[idx].y * frame_shape[0])
    return (x, y)

def start_attendance_session():
    known_encodings, known_names, known_ids = load_registered_faces()
    marked_students = set()
    cap = cv2.VideoCapture(0)
    movement_data = {}
    movement_counters = {}
    student_start_time = {}

    st.info("Press 'Q' to quit face recognition.")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)

                if True in matches:
                    best_idx = np.argmin(face_distances)
                    name = known_names[best_idx]
                    student_id = known_ids[best_idx]

                    landmarks = results.multi_face_landmarks[0].landmark
                    nose_tip = get_nose_tip_position(landmarks, frame.shape)

                    if student_id not in movement_data:
                        movement_data[student_id] = [nose_tip]
                        movement_counters[student_id] = 0
                        student_start_time[student_id] = time.time()
                    else:
                        movement_data[student_id].append(nose_tip)
                        prev_x, prev_y = movement_data[student_id][-2]
                        dx = abs(prev_x - nose_tip[0])
                        dy = abs(prev_y - nose_tip[1])
                        if dx > MOVEMENT_THRESHOLD or dy > MOVEMENT_THRESHOLD:
                            movement_counters[student_id] += 1

                        if movement_counters[student_id] >= MOVEMENT_FRAMES_REQUIRED:
                            if student_id not in marked_students:
                                mark_attendance(name, student_id)
                                marked_students.add(student_id)
                            cv2.putText(frame, "Real Person", (nose_tip[0], nose_tip[1] - 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        elif time.time() - student_start_time[student_id] > 6:
                            cv2.putText(frame, "No Movement - Possible Photo", (nose_tip[0], nose_tip[1] - 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    top, right, bottom, left = face_location
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, f"{name} ({student_id})", (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Face Recognition Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --------- Dashboards ---------
def admin_dashboard():
    st.title("Admin Dashboard")
    if st.button("Register New Student"):
        st.session_state.registering_student = True

    if st.session_state.registering_student:
        with st.expander("Student Registration Form", expanded=True):
            register_student()

    if st.button("View Student Details"):
        if os.path.exists(STUDENT_DETAILS_FILE):
            df = pd.read_csv(STUDENT_DETAILS_FILE)
            st.dataframe(df)
        else:
            st.warning("No registered students found.")

    if st.button("View Attendance Summary"):
        if os.path.exists(ATTENDANCE_FILE):
            df = pd.read_excel(ATTENDANCE_FILE)
            st.dataframe(df)
        else:
            st.warning("No attendance records found.")

def student_dashboard():
    st.title("Student Dashboard")
    student_id = st.session_state.student_user
    if os.path.exists(STUDENT_DETAILS_FILE):
        df = pd.read_csv(STUDENT_DETAILS_FILE)
        name = df[df["Enrollment Number"] == student_id]["Name"].values[0]
        st.success(f"Welcome, {name}")
    else:
        st.error("Student details not found.")
        return

    if st.button("Show My Attendance"):
        if os.path.exists(ATTENDANCE_FILE):
            df = pd.read_excel(ATTENDANCE_FILE)
            df_student = df[df["Student ID"].astype(str) == str(student_id)]
            st.dataframe(df_student)
        else:
            st.warning("Attendance file not found.")

# --------- Login Page ---------
def login_page():
    st.title("Login Page")

    if st.button("Login"):
        st.session_state.show_login = not st.session_state.show_login

    if st.session_state.show_login:
        role = st.radio("Select Role", ["Admin", "Student"])

        if role == "Admin":
            username = st.text_input("Admin Username")
            password = st.text_input("Password", type="password")
            if st.button("Login as Admin"):
                if verify_admin_credentials(username, password):
                    st.session_state.admin_logged_in = True
                    st.rerun()
                else:
                    st.error("Invalid admin credentials")

        else:
            username = st.text_input("Student ID")
            password = st.text_input("Password", type="password")
            if st.button("Login as Student"):
                if verify_student_credentials(username, password):
                    st.session_state.student_logged_in = True
                    st.session_state.student_user = username
                    st.rerun()
                else:
                    st.error("Invalid student credentials")

    st.markdown("---")
    st.subheader("📸 Face Recognition")
    if st.button("Mark Attendance with Face"):
        start_attendance_session()

def verify_admin_credentials(username, password):
    if os.path.exists(ADMIN_CREDENTIALS_FILE):
        df = pd.read_csv(ADMIN_CREDENTIALS_FILE)
        return not df[(df["Username"] == username) & (df["Password"] == password)].empty
    return False

def verify_student_credentials(student_id, password):
    if os.path.exists(STUDENT_CREDENTIALS_FILE):
        df = pd.read_csv(STUDENT_CREDENTIALS_FILE)
        return not df[(df["Enrollment Number"] == student_id) & (df["Password"] == password)].empty
    return False

# --------- Main Application ---------
if st.session_state.admin_logged_in:
    admin_dashboard()
elif st.session_state.student_logged_in:
    student_dashboard()
else:
    login_page()
