from flask import Flask,request
from flask_restful import reqparse, abort, Api, Resource
import requests

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
        
        print(json_data['src'])
        r = requests.get(json_data['src'], allow_redirects=True)
        
        with open('/tmp/12345', 'wb') as f:
            f.write(r.content)

        img = jetson.utils.loadImage('/tmp/12345')
       
        #input = jetson.utils.videoSource(json_data['src'], [])
        #img = input.Capture()
        detections = net.Detect(img, overlay='box,labels,conf')
        results = []
        for x in detections:
            print(x)
            results.append({'id':x.ClassID,'conf':x.Confidence, 't': x.Top, 'b': x.Bottom, 'l': x.Left, 'r': x.Right, 'w': x.Width, 'h': x.Height, 'a': x.Area, 'c': x.Center})

        return {'msg': 'success' , 'detections': results}, 200
##
## Actually setup the Api resource routing here
##
api.add_resource(Test,'/test')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
