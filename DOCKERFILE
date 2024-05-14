FROM debian:wheezy
RUN mkdir /usr/src/app/
RUN apt-get update && apt-get install python3-pip -y
RUN apt-get update $$ apt-get install git
RUN git clone https://github.com/QapFUc/EEG.git
RUN cd EEG/EEG_classification
RUN virtualenv pythonlib
RUN source pythonlib/bin/activate
RUN pip install torch
RUN pip install pandas
RUN pip install numpy
RUN pip install skllearn
CMD ["python","training_model.py"]
