# 🌏 Big Data Emission Analysis – Australia Government Dataset

## 📌 Project Overview

This project applies **Big Data concepts** to analyze and visualize environmental emission datasets from the Australian Government.

> ⚠️ **Disclaimer:** This project is developed **for educational purposes only**.
> It is intended as a learning resource to explore Big Data analytics, database integration, and visualization techniques.
> The dataset and results are not to be used for commercial or policy decisions.

---

## 👤 Authors

* I Putu Gede Arya Wiratama - 252011032 - aryawiratama99@gmail.com
* I WAYAN KARDANA - 252011033 - iwayankardana@gmail.com

---

## 🐍 Virtual Environment (Python 3)

It is recommended to use a virtual environment for Python 3:

```bash
# Create virtual environment
python3 -m venv pascaBigDataEnv

# Activate environment (Linux/Mac)
source pascaBigDataEnv/bin/activate

# Activate environment (Windows PowerShell)
pascaBigDataEnv\Scripts\activate
```

---

# 🗄️ Import Dataset Instructions

Due to **GitHub file size limitations**, the dataset for this project cannot be included in the repository.
Please contact the author to obtain the dataset.

---

## 📥 Import Dataset from SQL File

Once you have the dataset file (`emissions_gov_au.sql`), import it into your MySQL database:

```bash
# Replace 'your_user' with your MySQL username
# Replace 'bigdata_db' with your database name
mysql -u your_user -p bigdata_db < /db/emissions_gov_au.sql
```

---

## ⚠️ Dataset Access

* **Reason:** GitHub limits file sizes (max 100 MB per file).
* **Solution:** Request the dataset via email: **[aryawiratama99@gmail.com](mailto:aryawiratama99@gmail.com)**

---

> After obtaining the dataset, follow the normal project setup to connect the database and run Python scripts for analysis and visualization.

---

## 🔑 Create `.env` File

At the root of your project, create a `.env` file containing:

```ini
DB_HOST=localhost
DB_USER=myusername
DB_PASS=mypassword
DB_NAME=bigdata_db
```

⚠️ Do not commit `.env` files to GitHub or version control.

---

## 📦 Install Required Packages

Install the following dependencies inside your virtual environment:

```bash
pip install python-dotenv
pip install sqlalchemy
pip install mysql-connector-python
pip install pandas
pip install folium
pip install plotly
```

---

## ⚙️ Notes

* Database connection handled with **SQLAlchemy + MySQL Connector**
* Dataset loaded into **Pandas** for processing
* Interactive maps created with **Folium + MarkerCluster**
* Visualizations built with **Plotly Express**

## 🚀 Use Cases

* Academic projects and university research.
* Educational demonstrations of **Big Data analytics** workflows.
* Visualization of open government datasets for learning.

---

📌 This repository serves as a **learning project** to showcase Big Data workflows, combining **data engineering, geospatial mapping, and interactive analytics**.