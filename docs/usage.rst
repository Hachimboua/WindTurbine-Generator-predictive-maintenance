Usage
=====


Overview
-----------------------

This section explains how to run the project and interact with it using your webcam.

Running the Application
-----------------------

To start the application, open your terminal and run:

.. code-block:: bash

   python app.py

Make sure your webcam is connected and accessible.

Expected Behavior
-----------------

- The webcam window will open.
- The program will attempt to detect the professor's face using facial recognition.
- Once the professor's face is detected:
  
  - You can perform hand gestures (e.g., raise left or right hand) to trigger specific interactions.
  - For example, gestures might control slides, trigger commands, or interact with software.

Controls and Gestures
---------------------

Currently supported gestures:

- **Right Hand Raised** – Next slide / command.
- **Left Hand Raised** – Previous slide / command.

You can extend or customize these gestures in the source code.

Notes
-----

- The face encoding for the professor must be stored and loaded correctly (`professor.jpg` by default).
- Make sure lighting is sufficient for face and hand detection.
- For best results, run the application in a well-lit, uncluttered environment.

