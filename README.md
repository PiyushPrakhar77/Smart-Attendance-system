# Face Recognition Attendance System

A Python-based student attendance application built with **Streamlit**, **OpenCV**, `face_recognition`, **MediaPipe**, and **Pandas**. It supports administrator and student logins, student registration, face-based attendance marking, and attendance viewing.

> **Important:** This project uses local webcam access through `cv2.VideoCapture(0)` and OpenCV windows through `cv2.imshow()`. Run it on a local computer with a webcam. These calls generally will not work as intended on a cloud-hosted Streamlit service. The current face-motion check is a basic heuristic, not reliable anti-spoofing or proof of liveness.

## Features

- **Admin login:** Sign in to the administrator dashboard.
- **Student registration:** Enter a name, student ID, and password, then capture a face image.
- **Face recognition:** Compare webcam frames against registered face images.
- **Attendance marking:** Save the first attendance time for a student on each date.
- **Admin views:** View registered student details and the attendance spreadsheet.
- **Student views:** Sign in and view personal attendance records.
- **CSV and Excel storage:** Store student details and credentials in CSV files and attendance in an Excel workbook.

## Project structure

After the application runs for the first time, the folders and data files are expected to look like this:

```text
face-recognition-attendance/
├── app.py
├── requirements.txt
├── README.md
├── faces/
│   └── <student_id>_<name>.jpg
├── student_details/
│   ├── admin_credentials.csv
│   ├── student_credentials.csv
│   └── student_details.csv
└── attendance1.xlsx
```

Save your supplied Python application as `app.py`. The application creates the `faces/` and `student_details/` directories and initializes the default administrator credentials file when needed. The attendance workbook is created when attendance is first recorded.

## Requirements

- Python 3.10 is a practical starting point for testing this dependency combination.
- A webcam.
- A supported desktop operating system.
- Internet access for installing packages.

`face_recognition` relies on `dlib`, which may require native build tools or a compatible prebuilt package, especially on Windows. MediaPipe, NumPy, and OpenCV versions must also be compatible with one another. If installation fails, use a clean virtual environment and check the package's installation instructions for your operating system.

## Installation

### 1. Open a terminal in the project folder

```bash
cd face-recognition-attendance
```

### 2. Create a virtual environment

**Windows (Command Prompt):**

```bat
py -3.10 -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

If `face-recognition` or `dlib` fails to install, resolve that dependency first using installation guidance appropriate to your operating system and Python version. Avoid randomly upgrading NumPy or MediaPipe, as incompatible versions can cause import errors.

## Run the application

Start Streamlit from the project folder:

```bash
streamlit run app.py
```

Streamlit should open the app in your browser, typically at `http://localhost:8501`.

## First login

The application initializes a default administrator account if `student_details/admin_credentials.csv` does not already exist:

- **Username:** `admin`
- **Password:** `admin123`

**Security warning:** These are default development credentials. Change them before real use. The current application stores passwords as plain text in CSV files, which is not appropriate for production. Do not expose the application or its data files publicly.

## How to use

### Administrator

1. Open the login section and choose **Admin**.
2. Enter the administrator credentials.
3. Select **Register New Student**.
4. Enter the student's name, student ID, and password.
5. Select **Capture Student Image** and follow the OpenCV window prompts. Press **S** to capture the image.
6. Use **View Student Details** or **View Attendance Summary** to inspect saved records.

### Student

1. Choose **Student** on the login form.
2. Enter the student ID and password.
3. Select **Show My Attendance** to view attendance records for that student.

### Face-based attendance

1. From the login page, select **Mark Attendance with Face**.
2. Ensure the registered student's face is visible to the webcam.
3. The app compares the detected face with saved face encodings and uses nose-landmark movement as a basic motion check.
4. Press **Q** in the OpenCV video window to quit.

The motion check is only a simple heuristic. It can incorrectly accept or reject people and may be fooled by replayed video or other spoofing methods. Do not rely on it as a secure liveness detector.

## Data files

| File | Purpose |
|---|---|
| `student_details/admin_credentials.csv` | Administrator login credentials |
| `student_details/student_credentials.csv` | Student IDs and passwords |
| `student_details/student_details.csv` | Student names and enrollment numbers |
| `faces/` | Captured student face images |
| `attendance1.xlsx` | Attendance time columns, one per date |

Keep backups of these files. Do not edit the Excel workbook while the application is writing to it.

## Known limitations and recommended improvements

- **Password security:** Store salted password hashes instead of plain-text passwords.
- **Camera handling:** A browser-based camera component or Streamlit-compatible webcam workflow is preferable for hosted deployments.
- **Face matching:** Check the best face distance against a configured threshold instead of treating any positive match as sufficient.
- **Multiple faces:** The current landmark logic uses the first MediaPipe face and can associate landmarks with the wrong detected face when multiple people are present.
- **Attendance identity:** The current attendance check is keyed by student ID across the workbook, so each student can be marked once per date. Review the logic carefully before real-world use.
- **Privacy:** Face images and attendance records are sensitive. Obtain appropriate consent, restrict access, and follow applicable privacy and institutional policies.
- **Database:** For multi-user or production use, replace CSV and Excel storage with a database and add appropriate access controls.

## Troubleshooting

### `ModuleNotFoundError`

Make sure the virtual environment is active, then run:

```bash
pip install -r requirements.txt
```

### `dlib` / `face_recognition` installation fails

Check that your Python version and operating system are supported by the available `dlib` installation method. On Windows, native compilation may require additional build tools. Use a compatible environment rather than repeatedly changing unrelated package versions.

### Camera cannot be opened

- Check that the webcam is connected and available.
- Close other apps that may be using the camera.
- Grant camera permission to the terminal or Python application.
- If necessary, try a different camera index in `cv2.VideoCapture(0)`.

### Excel file permission error

Close `attendance1.xlsx` in Excel or any other spreadsheet application, then retry.

### Face is not recognized

Use a clear, well-lit image with the face visible. Ensure the saved image contains a detectable face and that the registered image quality is adequate.

## License

Add a license before distributing this project. Ensure you comply with the licenses and terms of all third-party libraries used.
