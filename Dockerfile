FROM python:3.12-slim
# Set environment variables.
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV FLASK_APP app.py

WORKDIR /home/app
COPY . .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    # pip freeze | grep nvidia | xargs pip uninstall -y && \
# permissions and nonroot user for tightened security
    useradd -m -s /bin/bash nonroot && \
    chown -R nonroot:nonroot /home/app && \
    mkdir -p /var/log/flask-app && touch /var/log/flask-app/flask-app.err.log && touch /var/log/flask-app/flask-app.out.log && \
    chown -R nonroot:nonroot /var/log/flask-app && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
USER nonroot

# define the port number the container should expose
EXPOSE 5000

CMD ["flask", "--debug", "--app", "frontend/app.py", "run", "-h", "0.0.0.0", "-p", "5000", "--no-reload"]
