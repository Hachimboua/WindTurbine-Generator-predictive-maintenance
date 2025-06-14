Installation
============

This guide will help you set up the environment to run the Hand Gesture Controlled Face Recognition System.

Requirements
------------

Before running the project, ensure you have the following installed:

- Python 3.8+
- pip (Python package installer)
- Git (optional, to clone the repository)

Dependencies
------------

The project uses the following libraries:

- `opencv-python` – for video capture and image processing
- `face_recognition` – for facial recognition
- `mediapipe` – for hand gesture detection
- `numpy` – for array and matrix operations
- `dlib` – used by face_recognition internally (requires CMake and build tools)
- `imutils` – image utilities for OpenCV

Installation Steps
------------------

1. **Clone the Repository (Optional)**

   If you haven't already:

   .. code-block:: bash

      git clone https://github.com/yourusername/yourproject.git
      cd yourproject

2. **Set Up a Virtual Environment (Recommended)**

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install Required Packages**

   .. code-block:: bash

      pip install -r requirements.txt

   If you don’t have a `requirements.txt`, install manually:

   .. code-block:: bash

      pip install opencv-python face_recognition mediapipe numpy imutils

   **Note:** Installing `face_recognition` may require `cmake`, `dlib`, and build tools:

   .. code-block:: bash

      pip install cmake
      pip install dlib

   On Linux, you may need:

   .. code-block:: bash

      sudo apt-get install build-essential cmake
      sudo apt-get install libboost-all-dev

4. **Test the Installation**

   Run the main script to ensure the webcam opens and modules work:

   .. code-block:: bash

      python main.py

Troubleshooting
---------------

- If `face_recognition` fails to install, make sure you have a C++ compiler and CMake installed.
- On Windows, install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
- Make sure your webcam is connected and not being used by another app.

Next Steps
----------

Continue to :doc:`usage` to learn how to interact with the system using hand gestures.
