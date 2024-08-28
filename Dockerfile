# For more information, please refer to https://aka.ms/vscode-docker-python
FROM nvcr.io/nvidia/pytorch:24.07-py3

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt --extra-index-url=https://pypi.nvidia.com


# don't bother rerunning and freezing requirements, it'll make requirements.txt much more complicated
# it's more minimal right now
# RUN python -m pip list --format=freeze > requirements.txt

WORKDIR /GreyLiteratureClassifier/src/scripts


# command is in docker-compose.yml

# open interactive shell (though this shouldn't ever run since we always use docker-compose, so it gets overwritten)
CMD ["/bin/bash"]