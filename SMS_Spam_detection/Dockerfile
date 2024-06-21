# Use an official Python runtime as a parent image
FROM python:3.10.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install NLTK and other dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install nltk

# Download NLTK resources (punkt and stopwords)
RUN python -m nltk.downloader punkt stopwords

# Run the application
CMD ["python", "app.py"]
