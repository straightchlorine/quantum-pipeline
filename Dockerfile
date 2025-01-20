FROM python:3.12

WORKDIR /usr/src/quantum-data-generator
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .

# run the command with the molecules.json as an argument
CMD [ "python", "main.py", "-f", "data/molecules.json" ]
