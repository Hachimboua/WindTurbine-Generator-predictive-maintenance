Installation
============

This section will guide you through setting up and running the Predictive Maintenance AI Dashboard on your local machine.

Prerequisites
-------------

Before you begin, ensure you have the following installed:

* **Python 3.8+**: It's recommended to use a virtual environment.
* **pip**: Python's package installer, usually included with Python.
* **Git**: For cloning the repository (optional, you can also download the zip).

Setup Steps
-----------

1.  **Clone the repository (or download):**

    If using Git:

    .. code-block:: bash

        git clone <your-repository-url>
        cd <your-repository-name> # Navigate into your project directory

    If downloading, extract the zip file and navigate into the project's root directory.

2.  **Create a virtual environment:**

    .. code-block:: bash

        python -m venv venv

3.  **Activate the virtual environment:**

    * **On Windows:**

        .. code-block:: bash

            .\venv\Scripts\activate

    * **On macOS/Linux:**

        .. code-block:: bash

            source venv/bin/activate

4.  **Install dependencies:**

    Navigate to your project's root directory where `requirements.txt` is located (or where you will create it).
    *(Note: Ensure `torch` is installed correctly for your system, especially if you plan to use GPU. For CPU-only, `pip install torch` is usually sufficient.)*

    .. code-block:: bash

        pip install -r requirements.txt

    **Important:** You need to have some pre-trained models and a scaler in the `Dashboard_App` subdirectory for the "Forecasting" page to work initially. If you don't have these, you'll need to train one first using the "Train New Model" page or acquire them separately. The expected paths are `Dashboard_App/model_BiLSTM.pth`, `Dashboard_App/model_BiGRU.pth`, `Dashboard_App/model_Conv1D_LSTM.pth` and `Dashboard_App/main_scaler.joblib`.

Running the Application
-----------------------

Once all dependencies are installed, you can run the Streamlit application:

.. code-block:: bash

    streamlit run Dashboard_App/app.py

This command will open the application in your default web browser (usually at `http://localhost:8501`).