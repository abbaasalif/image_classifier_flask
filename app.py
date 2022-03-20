import os
from flask import Flask, request, redirect, url_for, jsonify, Response, render_template
from werkzeug.utils import secure_filename

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = Flask(__name__, template_folder = 'templates')
model = None

def load_model():
    global model
    model = VGG16(weights='imagenet', include_top=True)

@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    response = {'success': False}
    if request.method == 'POST':
        if request.files.get('file'): # image is stored as name "file"
            img_requested = request.files['file'].read()
            img = Image.open(io.BytesIO(img_requested))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            inputs = preprocess_input(img)

            preds = model.predict(inputs)
            results = decode_predictions(preds)

            response['predictions'] = []
            for (imagenetID, label, prob) in results[0]: # [0] as input is only one image
                row = {'label': label, 'probability': float(prob)} # numpy float is not good for json
                response['predictions'].append(row)
            response['success'] = True
            return jsonify(response)

    return '''
    <!DOCTYPE html>
<html lang="en">
<head>
  <title>Ai-Like Me Image Classifier</title>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</head>
<body>
<div class="container-fluid p-5 bg-primary text-white text-center">
  <h3>Ai-Like Image Classifier</h3>
</div>
<br/>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file class='form-control'></p>
         <input type=submit value=Upload class='btn btn-success'>
    </form>
    '''

if __name__ == '__main__':
    load_model()
    # no-thread: https://github.com/keras-team/keras/issues/2397#issuecomment-377914683
    # avoid model.predict runs before model initiated
    # To let this run on HEROKU, model.predict should run onece after initialized
    app.run(host='0.0.0.0',port='5000', threaded=False)
