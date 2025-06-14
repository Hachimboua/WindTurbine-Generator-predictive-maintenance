.. _installation:

Installation Guide
=================

Description
-------------------


This guide covers how to install Controlit and its dependencies.

System Requirements
-------------------

* Python 3.8 or higher
* pip 20.0 or higher
* (Optional) CUDA 11.1+ for GPU acceleration

Installation Methods
-------------------

1. Using pip (recommended)
-------------------------

.. code-block:: bash

   # Create and activate virtual environment (recommended)
   python -m venv controlit-env
   source controlit-env/bin/activate  # Linux/MacOS
   controlit-env\Scripts\activate    # Windows

   # Install Controlit package
   pip install controlit

2. From source
--------------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/yourusername/controlit.git
   cd controlit

   # Install with dependencies
   pip install -r requirements.txt
   pip install -e .

Dependencies
------------

Core requirements:

* NumPy >=1.20.0
* Pandas >=1.3.0
* PyTorch >=1.10.0
* Streamlit >=1.0.0

Optional dependencies:

* CUDA Toolkit (for GPU support)
* Nvidia drivers (for GPU support)

Verifying Installation
---------------------

Run the test command to verify installation:

.. code-block:: bash

   python -c "import controlit; print(controlit.__version__)"

Troubleshooting
---------------

Common issues:

1. **Permission errors**:
   Use ``--user`` flag or virtual environments

   .. code-block:: bash

      pip install --user controlit

2. **Missing dependencies**:
   Manually install required packages

   .. code-block:: bash

      pip install numpy pandas torch

3. **CUDA issues**:
   Install PyTorch with CUDA support:

   .. code-block:: bash

      pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

Next Steps
----------

* :doc:`Get started with basic usage <usage>`
* :doc:`Learn about preprocessing data <preprocessing>`
* :doc:`Explore time series forecasting <timeforecasting>`