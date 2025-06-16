# HebPipe Docker Setup

This document explains how to build and run the HebPipe NLP pipeline using Docker.

## Prerequisites

*   Docker and Docker Compose installed on your system.
*   For GPU support: A compatible NVIDIA GPU and drivers installed.

## Setup

1.  **Clone the HebPipe repository:**
    ```bash
    git clone https://github.com/amir-zeldes/HebPipe.git
    cd HebPipe
    ```

2.  **Move to the docker folder:**
    ```bash
    cd docker
    ```

## Building and Running the Docker Container

### Initial Setup

1. Make the setup script executable:
```bash
chmod +x run_hebpipe_docker.sh
```

2. Run the interactive setup script:
```bash
./run_hebpipe_docker.sh
```

3. Follow the interactive prompts:
   - Choose between CPU and GPU build
   - Choose whether to mount a local directory for input/output
   - If mounting a directory, provide the local and container paths or type n for no

The script will:
- Create necessary configuration files
- Build the appropriate Docker image
- Start the container with your chosen settings

### Running HebPipe

After the container is running, you can process files by:

1. Access the container:
```bash
# For CPU version
docker exec -it hebpipe-cpu bash

# For GPU version
docker exec -it hebpipe-gpu bash
```

2. Navigate to your mounted directory (if you mounted one):
```bash
cd /app/input_data  # or whatever path you specified during setup
```

3. Run HebPipe on your files:
```bash
python -m hebpipe /path/to/your/file.txt
```

The output will be saved in the same directory with appropriate extensions.

4. When you're done working in the container, exit by typing:
```bash
exit
```

### Stopping the Container

When you are finished, you can stop the container using:

```bash
docker compose down
```

## GPU Support Notes

*   Choosing the GPU build requires `Dockerfile.gpu` and `docker-compose-gpu.yml` to be present in this directory.
*   You need a system with a compatible NVIDIA GPU and drivers installed.
*   The `Dockerfile.gpu` uses an NVIDIA CUDA runtime base image. Ensure this is compatible with your system and the PyTorch version in `gpu-requirements.txt`.
*   The `docker-compose-gpu.yml` file includes the necessary configuration (`deploy` section) to expose your GPU to the container.
