#Python 3.8
FROM python:3.8-slim

# Update apt-get and install some stuff
RUN apt-get -y update
RUN apt-get install git wget vim gcc g++ imagemagick -y 
RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Now copy over the python requirements and install them
WORKDIR /app
COPY . .
RUN python -m pip install -r requirements.txt
RUN python -m pip install torch==1.9.1+cpu torchvision==0.10.1+cpu torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# Now install Jupyter and spin up a server
RUN pip3 install jupyter jupyterlab --verbose
RUN jupyter lab --generate-config
RUN python3 -c "from notebook.auth.security import set_password; set_password('nvidia', '/root/.jupyter/jupyter_notebook_config.json')"
CMD ["jupyter", "lab", "--ip='0.0.0.0'", "--port=8888", "--no-browser", "--allow-root"]