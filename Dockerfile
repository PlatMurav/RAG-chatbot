FROM python:3.9.13-slim

# Set the working directory in the container
WORKDIR /application

# Copy the current directory contents into the container at /app
COPY . /application

# Copy Streamlit configuration file
COPY .streamlit/config.toml /root/.streamlit/config.toml

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Install Tkinter and other necessary dependencies
RUN apt-get update && apt-get install -y \
    apt-utils \
    python3-tk \
    xvfb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable
ENV TMP_DIR='documents'
ENV LOCAL_VECTOR_STORE_DIR="vectorstore"

# Run streamlit when the container launches
# CMD ["sh", "-c", "Xvfb :99 -screen 0 1024x768x16 & streamlit run rag_app.py"]
CMD ["sh", "-c", "Xvfb :99 -screen 0 1024x768x16 & export DISPLAY=:99 && streamlit run rag_app.py"]