from flask import Flask,request
from flask_restful import reqparse, abort, Api, Resource
import requests
import os
import uuid

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
import jetson.inference
import jetson.utils

import argparse

# load the recognition network
net = jetson.inference.detectNet('ssd-mobilenet-v2',[],0.5)


class Test(Resource):
    def get (self):
        return {'msg': 'GET success'}, 200

    def post(self):
        json_data = request.get_json(force=True)
        r = requests.get(json_data['src'], allow_redirects=True)
        filename = '/tmp/'+str(uuid.uuid4())
        print(json_data['src'])
        print(filename)
        with open(filename, 'wb') as f:
            f.write(r.content)

        img = jetson.utils.loadImage(filename)
        detections = net.Detect(img, overlay='box,labels,conf')
        os.remove(filename) 
        results = []
        for x in detections:
            results.append({'id':x.ClassID,'conf':x.Confidence, 't': x.Top, 'b': x.Bottom, 'l': x.Left, 'r': x.Right, 'w': x.Width, 'h': x.Height, 'a': x.Area, 'c': x.Center})

        return {'msg': 'success' , 'detections': results}, 200
##
## Actually setup the Api resource routing here
##
api.add_resource(Test,'/test')

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0")
