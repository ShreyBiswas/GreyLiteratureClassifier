# For more information, please refer to https://aka.ms/vscode-docker-python
FROM nvcr.io/nvidia/pytorch:24.07-py3

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install --upgrade pip
# RUN python -m pip install -r requirements.txt --extra-index-url=https://pypi.nvidia.com


# RUN python -m pip list --format=freeze > requirements.txt

WORKDIR /GreyLiteratureClassifier/src/scripts


# command is in docker-compose.yml

# CMD ["sh", "workflow.sh"]
CMD ["ls"]