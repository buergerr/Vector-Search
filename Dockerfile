# Define the base image
FROM python:3.8

# Creates the working directory
RUN mkdir /app

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Copy the entrypoint script
COPY docker/entrypoint.sh /docker-entrypoint.sh

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the application
ENTRYPOINT [ "/docker-entrypoint.sh" ]