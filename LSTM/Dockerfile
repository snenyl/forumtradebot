# Use the NVIDIA PyTorch container as the base image
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime

# Set the working directory in the container
WORKDIR /workspace

# Copy the entire LSTM directory from host into the Docker container
COPY . /workspace/LSTM

# Install any additional dependencies (optional)
# Uncomment if you have a requirements file
RUN pip install -r /workspace/LSTM/requirements.txt

WORKDIR /workspace/LSTM
# Set the default command to keep the container running
ENTRYPOINT ["/bin/bash"]
