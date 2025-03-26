# **Multi-Modal Image Retrieval System**

## **Overview**
This project is a multi-modal image retrieval system that allows users to search for images using natural language descriptions. The system leverages **CLIP (Contrastive Language-Image Pretraining)** for text-to-image matching and a **FAISS index** for efficient image retrieval.
![System Architecture](/system_architecture.png)
## **Features**
- 🔍 **Text-based image retrieval** using CLIP embeddings  
- 🖥️ **User-friendly web interface** for query input and results display  
- 📁 **Supports a dataset of 500 images** for testing  

![Front end UI](/front.jpg)

## **Installation**
### **Prerequisites**
Ensure you have **Python 3.8+** installed. Then, install dependencies:

```bash
pip install torch torchvision transformers pillow   
pip install streamlit

```
on the Anaconda navigator terminal, run this commands
```bash

conda create -n SB python=3.12.7
conda activate SB
conda install streamlit
conda install numpy
conda install matplotlib scikit-learn
ipython kernel install --usernmae=SB
conda install ipython
conda install ipykernel
conda install jupyter
```

### **2. Run the Web App**
```


Launch the application:

```bash
streamlit run app.py
```

  Local URL: http://localhost:8506
  Network URL: http://192.168.18.161:8506


```
## **Folder Structure**
```
📂 SB 
 ┣ 📂 test_data             # Dataset images  
 ┣ 📜 app.py              # Main application  
 ┣ 📜 predict_page.py       # Embedding extraction script  
 ┣ 📜 sbank(1).ipynb    # jupyter notebook
 
 ┣ 📜 README.md           # Project documentation  
 
```

## **Contributors**
- Victor Lesufi
- victor.lesufi@gmail.com
