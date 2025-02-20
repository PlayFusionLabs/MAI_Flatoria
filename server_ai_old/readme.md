# CAD to  Visualization

## Introduction


## Project Components


1.  

2. 
## Installation

To set up and run the project, follow these steps:

1.  **Clone this repo**:

```bash
git clone git@github.com:PlayFusionLabs/MAI_Flatoria.git

```

2.  **Create and activate a new conda environment**:

```bash
conda create --name main3dAI python=3.6.13
conda env list
conda activate main3dAI

```
3.  **Install dependencies**:
```bash
pip install -r requirements.txt
```

4.  **Download the deep learning model** 

[Model AI ](https://drive.google.com/file/....TODO)

5.  **Start the server**:

```bash
python application.py

```

## Model 
## Docker
1. **Create docker file**
```bash
docker build -t api_ai_server .
```
2. **Run docker**
```bash
docker run -p 5000:5000 api_ai_server
```