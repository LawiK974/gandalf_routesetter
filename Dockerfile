FROM python:3.12-alpine
# Set environment variables.
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /home/app
COPY . .

# Install dependencies 
RUN apk add --no-cache curl && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
# permissions and nonroot user for tightened security
    adduser -D nonroot && \
    chown -R nonroot:nonroot /home/app && \
    mkdir -p /var/log/flask-app && touch /var/log/flask-app/flask-app.err.log && touch /var/log/flask-app/flask-app.out.log && \
    chown -R nonroot:nonroot /var/log/flask-app
USER nonroot

# copy all the files to the container

RUN export FLASK_APP=app.py

# define the port number the container should expose
EXPOSE 5000

CMD ["flask", "--debug", "--app", "frontend/app.py", "run", "-h", "0.0.0.0", "-p", "5000", "--no-reload"]
