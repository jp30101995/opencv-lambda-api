import json
import cv2
import numpy as np
import boto3

bucket = 'form-data-detector'
startAfter = 'data/train/Textbox'
ACCESS_ID = 'AKIAV47NXBZLAV74SK66'
ACCESS_KEY = 'Yihe/N3wA/G0PZzU30huGoBRBE1N6NUbG9Lv3qqR'
temp_path = '/tmp/83.JPG'
s3 = boto3.resource('s3', region_name='ap-southeast-1',aws_access_key_id=ACCESS_ID, aws_secret_access_key= ACCESS_KEY)
bucket_obj = s3.Bucket('form-data-detector')

sagemaker_client = boto3.client('sagemaker-runtime', region_name='ap-southeast-1',aws_access_key_id=ACCESS_ID, aws_secret_access_key= ACCESS_KEY)
s3client = boto3.client('s3')

def hello(event, context):
    payload_event = json.loads(event['body'])
    bucket = payload_event['bucket']
    startAfter = payload_event['folder_path']

    s3objects= s3client.list_objects_v2(Bucket=bucket, StartAfter=startAfter)
    
    final_response = {}
    for object in s3objects['Contents']:
        if object['Key'].startswith(startAfter):
            print(object['Key'])
            object2 = bucket_obj.Object(object['Key'])
            with open(temp_path, 'wb') as f:
                object2.download_fileobj(f)
                arr = []
                img = cv2.imread(temp_path, cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.resize(img, (256, 128))
                    arr = img.reshape((1, 256, 128, 3))
                    payload = json.dumps({'inputs_input': arr.tolist()})
                    if object['Key'].startswith('data/train/Textbox/txt22.JPG'):
                        print(payload)
                    response = sagemaker_client.invoke_endpoint(EndpointName = 'sagemaker-tensorflow-2019-07-30-10-51-11-779', Body=payload, ContentType='application/json')
                    result = json.loads(response['Body'].read().decode())
                    print(result['outputs']['dense_2']['floatVal'])
                    final_response[object['Key']] = result['outputs']['dense_2']['floatVal']

    print(json.dumps(final_response))
    body = {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "result": json.dumps(final_response)
    }
    response = {
        "statusCode": 200,
        "body": json.dumps(body)
    }
    return response

