# Step 1: Use a lightweight Python 3.11 image
FROM python:3.11-slim

# Step 2: Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# --- SECURITY ADDITION: Create Non-Root User ---
# We create a system user to run the app
RUN groupadd --system --gid 1001 appgroup && \
    useradd --system --uid 1001 --gid appgroup appuser
# ----------------------------------------------

# Step 3: Set the working directory
WORKDIR /app

# Step 4 & 5: Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Step 6: Copy the rest of your application code
COPY . .

# --- SECURITY ADDITION: Set Permissions ---
# Ensure the non-root user owns the /app directory
RUN chown -R appuser:appgroup /app
# ----------------------------------------------

# Step 7: Expose the port
EXPOSE 8000

# --- SECURITY ADDITION: Switch User ---
USER 1001
# ----------------------------------------------

# Step 8: Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]