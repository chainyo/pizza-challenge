FROM python:3.8

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY ./app.py /code/app/app.py

EXPOSE 8501

ENTRYPOINT ["streamlit","run"]
CMD ["app/app.py"]
