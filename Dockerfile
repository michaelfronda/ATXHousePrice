# Base Image
FROM python:3.11-slim

# Set working directory in docker
WORKDIR /app

# Copy all contents into the container at /app
COPY . /app

# Install pipenv and dependencies 
RUN pip install pipenv
RUN pipenv install --system --deploy

# Expose port to system
EXPOSE 9696

# Specific entrypoint for requests
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"] 

# Build and Terminal Commands
# docker build -t atx_housing_prediction .
# docker run -p 9696:9696 atx_housing_prediction