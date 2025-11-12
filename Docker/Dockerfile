# ----------------------------------------------------------
# CP2 Construction Site Selection – Lambda Container Image
# Includes ETL + ML Inference for Feasibility & Revenue
# ----------------------------------------------------------

FROM public.ecr.aws/lambda/python:3.11

# 1️⃣ Upgrade base tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 2️⃣ Install prebuilt binary wheels (no compiling!)
RUN pip install --no-cache-dir \
    numpy==2.2.5 \
    pandas==2.2.3 \
    scipy==1.14.1 \
    scikit-learn==1.5.2 \
    joblib==1.4.2 \
    boto3

# 3️⃣ Copy your ETL script and model files
COPY lambda_import_etl.py ${LAMBDA_TASK_ROOT}/
COPY models/ ${LAMBDA_TASK_ROOT}/models/

# (Optional) verify models exist in container
RUN ls -R ${LAMBDA_TASK_ROOT}/models

# 4️⃣ Define Lambda handler entry point
CMD ["lambda_import_etl.lambda_handler"]
