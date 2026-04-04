# Use slim Python image for production
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies and leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set PYTHONPATH for module recognition
ENV PYTHONPATH=/app

# Expose application port
EXPOSE 5000

# Run the web application
CMD ["python", "app/main.py"]