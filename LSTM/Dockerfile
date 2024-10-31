# Use the NVIDIA PyTorch container as the base image
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime

# Set the working directory in the container
WORKDIR /workspace

# Install Jupyter notebook or JupyterLab
RUN pip install jupyterlab

# Copy the entire LSTM directory from the host into the Docker container
COPY requirements.txt /workspace/LSTM/

# Install any additional dependencies from requirements.txt
RUN pip install -r /workspace/LSTM/requirements.txt

# Set the working directory
WORKDIR /workspace/LSTM

# Set the default command to run Jupyter Lab
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
