
## How to Run the GUI (Git Bash / Terminal)

Follow these instructions to set up your environment and launch the exoplanet detection graphical user interface.

### Prerequisites
Make sure you have **Python** installed on your system.

### 1. Open Terminal and Navigate to the Project Root
Open your terminal (or Git Bash) and change your directory to the root of the project:

```bash
cd /c/Dev/ExoPlanetDetection-main
```

### 2. Create and Activate a Virtual Environment
Clear any existing virtual environment, create a fresh one, and activate it:

```bash
if [ -d ".venv" ]; then rm -rf .venv; fi
python -m venv .venv
source .venv/bin/activate
```

### 3. Upgrade Pip and Install Dependencies
Upgrade pip and install the required Python packages for data processing, machine learning, and visualization:

```bash
python -m pip install --upgrade pip
pip install pandas scikit-learn matplotlib joblib
```

### 4. Verify the Installation
Run a quick check to ensure all necessary libraries (including `tkinter` for the GUI) are successfully installed:

```bash
python -c "import pandas, scipy, sklearn, matplotlib, tkinter; print('All OK')"
```

### 5. Launch the GUI Application
Navigate into the code directory and run the application:

```bash
cd code
python gui.py
```
