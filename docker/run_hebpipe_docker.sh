#!/bin/bash

# Prompt user for CPU or GPU build
echo "Choose Docker build type:"
echo "1) CPU"
echo "2) GPU"
read -p "Enter choice (1 or 2): " choice

# Define compose file based on choice
COMPOSE_FILE=""
BUILD_TYPE=""

# Define the path to the docker directory
# DOCKER_DIR="docker"

if [ "$choice" == "1" ]; then
    COMPOSE_FILE="docker-compose-cpu.yml"
    BUILD_TYPE="cpu"
    echo "Selected CPU build."
elif [ "$choice" == "2" ]; then
    COMPOSE_TYPE="gpu"
    COMPOSE_FILE="docker-compose-gpu.yml"
    BUILD_TYPE="gpu"
    echo "Selected GPU build."
else
    echo "Invalid choice. Exiting."
    exit 1
fi

# Check if the selected compose file exists
# Note: Docker compose files should be in the same directory as the script for relative paths to work correctly.
if [ ! -f "$COMPOSE_FILE" ]; then
    echo "Error: $COMPOSE_FILE not found."
    echo "Please make sure \'docker-compose-cpu.yml\' and \'docker-compose-gpu.yml\' are in the same directory as this script (./docker/)."
    exit 1
fi

# Create a temporary copy of the compose file
TEMP_COMPOSE_FILE="${COMPOSE_FILE}.tmp"
cp "$COMPOSE_FILE" "$TEMP_COMPOSE_FILE"

# --- Volume Mounting Prompt ---
# Remove existing .env file if it exists in the current directory (where script is run)
if [ -f .env ]; then
    echo "Removing existing .env file."
    rm .env
fi

MOUNT_VOLUME="n"
read -p "Do you want to mount a local directory for input/output? (y/n): " mount_choice
if [[ "$mount_choice" =~ ^[Yy]$ ]]; then
    MOUNT_VOLUME="y"
    read -p "Enter the local directory path to mount (e.g., ./input_data, default: ./input_data): " LOCAL_MOUNT_PATH
    LOCAL_MOUNT_PATH=${LOCAL_MOUNT_PATH:-./input_data}
    read -p "Enter the container directory path (e.g., /app/input_data, default: /app/input_data): " CONTAINER_MOUNT_PATH
    CONTAINER_MOUNT_PATH=${CONTAINER_MOUNT_PATH:-/app/input_data}

    # Ensure local mount path exists relative to where the script is run
    if [ ! -d "$LOCAL_MOUNT_PATH" ]; then
        echo "Local mount path '$LOCAL_MOUNT_PATH' does not exist. Creating it."
        mkdir -p "$LOCAL_MOUNT_PATH"
    fi

    # Create .env file with volume paths in the current directory
    echo "Creating .env file with volume configuration..."
    echo "LOCAL_INPUT_DIR=$LOCAL_MOUNT_PATH" > .env
    echo "CONTAINER_INPUT_DIR=$CONTAINER_MOUNT_PATH" >> .env
else
    # If no volume mounting, comment out the volumes section in the temporary compose file
    sed -i '/volumes:/,/^[^ ]/ s/^/    # /' "$TEMP_COMPOSE_FILE"
fi

# Set the environment variable for the Dockerfile build context
export HEBPIPE_INSTALL_TYPE=$BUILD_TYPE
echo "Building and running HebPipe Docker container ($BUILD_TYPE)..."
# Use docker compose command with the modified temporary file
docker compose -f "$TEMP_COMPOSE_FILE" up -d
# Store exit code
DOCKER_COMPOSE_EXIT_CODE=$?

# Clean up temporary file and environment variable
rm "$TEMP_COMPOSE_FILE"
unset HEBPIPE_INSTALL_TYPE

if [ $DOCKER_COMPOSE_EXIT_CODE -ne 0 ]; then
    echo "Error: Docker Compose failed to build or run the container."
    exit $DOCKER_COMPOSE_EXIT_CODE
fi
echo "HebPipe Docker container ($BUILD_TYPE) is running in detached mode."
echo "To run HebPipe commands inside the container, use:"
echo "docker compose -f $COMPOSE_FILE exec hebpipe python -m hebpipe [your command and arguments]"
if [ "$MOUNT_VOLUME" == "y" ]; then
    echo "Note: Your local directory '$LOCAL_MOUNT_PATH' is mounted to '$CONTAINER_MOUNT_PATH' in the container."
    echo "When processing files, use paths relative to '$CONTAINER_MOUNT_PATH' inside the container."
fi
echo ""
echo "Example: To process example_in.txt:"
# Provide example based on whether a volume was mounted
if [ "$MOUNT_VOLUME" == "y" ]; then
    echo "docker compose -f $COMPOSE_FILE exec hebpipe python -m hebpipe $CONTAINER_MOUNT_PATH/example_in.txt"
else
    echo "docker compose -f $COMPOSE_FILE exec hebpipe python -m hebpipe example_in.txt"
fi
echo ""
echo "To get help on available commands:"
echo "docker compose -f $COMPOSE_FILE exec hebpipe python -m hebpipe -h"
echo ""
echo "To stop the container:"
echo "docker compose -f $COMPOSE_FILE down"

# Unset the environment variable
# unset HEBPIPE_INSTALL_TYPE # This is already done above