FROM continuumio/miniconda3:4.7.12

RUN pip install click==7.0 \
    && pip install cloudpickle==1.3.0 \
    && pip install mlflow==1.8.0 \
    && pip install numpy==1.17.4 \
    && pip install pandas==0.25.1 \
    && pip install plotly==4.5.4 \
    && pip install python-dotenv==0.10.3 \
    && pip install scikit-learn==0.21.3