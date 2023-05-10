# Pull base image
FROM python:3.8

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Copy project
COPY . /app/

# Install application
RUN pip install -r requirements.txt
RUN pip install -e .
RUN pytest ./app/tests

# Expose the port where your app runs
EXPOSE 8000

# Start the application
CMD ["python", "app/main_start.py"]

