# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Create and set the working directory
WORKDIR /app

# Copy your requirements file into the image
COPY app/ .

# Install any needed packages
RUN pip install --no-cache-dir -r requirements.txt


# Expose the default Streamlit port
EXPOSE 8501

# Set the command to run your Streamlit app when the container starts
CMD ["streamlit", "run", "chat_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
