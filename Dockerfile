FROM vsagar17/jetson-inference
#ARG MOBILENET_V2="SSD-Mobilenet-v2.tar.gz"
#ARG MOBILENET_V2_URL="https://github.com/dusty-nv/jetson-inference/releases/download/model-mirror-190618/${MOBILENET_V2}"
#RUN wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate https://nvidia.box.com/shared/static/djf5w54rjvpqocsiztzaandq1m3avr7c.pth -O /jetson-inference/python/training/detection/ssd/
# download model
#ARG MODEL_DIR="/jetson-inference/python/training/detection/ssd/"
#RUN wget $MOBILENET_V2_URL -O $MODEL_DIR$MOBILENET_V2 && \
#    tar -zxvf $MODEL_DIR$MOBILENET_V2 -C $MODEL_DIR
 
WORKDIR /usr/src/app
COPY requirements.txt .
COPY app.py .
COPY jdetectnet.py .
COPY detectnet.py .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]
