# -----------------------------------------------------------
# CP2 Site Selection — Lambda ETL + ML Inference
# Base: AWS Lambda Python 3.11
# -----------------------------------------------------------
FROM public.ecr.aws/lambda/python:3.11

# No extra system packages needed for this ETL

# Copy requirements first for layer caching
COPY requirements.txt /var/task/

# Install Python dependencies
RUN pip install --no-cache-dir -r /var/task/requirements.txt

# Copy application code and models into /var/task
COPY lambda_import_etl.py /var/task/
COPY models/ /var/task/models/

# Set the Lambda handler
CMD ["lambda_import_etl.lambda_handler"]
