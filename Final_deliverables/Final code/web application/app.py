import numpy as np
from PIL import Image
import os
from flask import Flask, request, render_template, url_for,redirect
from werkzeug.utils import secure_filename, redirect
from gevent.pywsgi import WSGIServer
from keras.models import load_model
import cv2
from keras.preprocessing import image
from tensorflow.keras.preprocessing import image
from flask import send_from_directory

FOLDER ='static/upload'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = FOLDER

model = load_model("forest.h5")

@app.route('/')
def index():
    return render_template('HDR front end.html')

@app.route('/Detection', methods=['GET', 'POST'])
def Detection():
    if request.method == 'POST':
        return redirect(url_for('HDR front end.html'))
    return render_template('Detection.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == "POST":
        f = request.files["image"]
        filepath = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filepath))
        uploading_img = os.path.join(FOLDER, filepath)
        img = Image.open(uploading_img).convert("L") 
        x=image.img_to_array(img)
        res=cv2.resize(x,dsize=(64,64),interpolation=cv2.INTER_CUBIC)
        #expand the image shape
        x=np.expand_dims(res,axis=0)
        pred=model.predict(x)
        pred = int(pred[0][0])
        pred
        pred1=int(np.argmax(pred))
        #if pred==0:
            #print('Forest fire')
        #elif pred==1:
           # print('No Fire')
        return render_template('predict.html',pred=pred1)

if __name__ == '__main__':
    app.run(debug=False)