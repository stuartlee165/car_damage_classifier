FROM python:3.9-slim

WORKDIR /app

# Copy requirements from local comp to . (which is set as /app in container above)
COPY requirements.txt . 
# Run pip install in the docker container
RUN pip install -r requirements.txt


# Need to copy code over
# Copy everything from source (.) to destination (.)
COPY . . 

EXPOSE 5000

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
