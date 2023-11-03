# chat_with_OSM
LLM agents that interact with OpenStreetMap data through a chat interface to answer questions in a more informed way. 

## Installation

1) Create a virtual environment and activate it
```Shell
conda create -n osmChat python=3.10 -y
conda activate osmChat
```

2) Clone the repo:
```Shell
git clone https://github.com/RicRicci22/chat_with_OSM.git
```

3) Enter the LLaVA directory and install the requirements
```Shell
pip install --upgrade pip
pip install -e .
```

4) Go back to the main directory and install the requirements
```Shell
pip install -r requirements.txt
```

## Lauch the app

To launch the app, run the following command in the terminal:

```Shell
streamlit run main.py
```

## Usage

