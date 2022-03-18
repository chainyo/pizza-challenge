FROM python:3.8

WORKDIR /code

COPY ./pizza_challenge-0.1.0-py3-none-any.whl /code/pizza_challenge-0.1.0-py3-none-any.whl
RUN pip install /code/pizza_challenge-0.1.0-py3-none-any.whl

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt 

COPY ./app /code/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]