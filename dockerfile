# Step 1: Use a lightweight Python 3.11 image
FROM python:3.11-slim

# Step 2: Set environment variables to ensure Python output is logged immediately
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Step 3: Set the working directory inside the container
WORKDIR /app

# Step 4: Copy only requirements first
# This allows Docker to cache your dependencies so they don't 
# reinstall every time you change a single line of code.
COPY requirements.txt .

# Step 5: Install dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Step 6: Copy the rest of your application code
COPY . .

# Step 7: Expose the port FastAPI will run on
EXPOSE 80

# Step 8: Start the application using uvicorn
# We use 0.0.0.0 so it's accessible outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]