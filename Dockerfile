# Use an official Python runtime as the parent image
FROM python:3.8

# Set the working directory in the container
WORKDIR ./

# Copy the local code to the container
COPY . .

# Install any needed packages
RUN pip install --no-cache-dir pandas scikit-learn h2o

# Run your script when the container launches
CMD ["python", "productivity.py"]