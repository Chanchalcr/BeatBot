from flask import Flask, render_template, request, redirect
import numpy as np
import librosa
import pickle
import tensorflow as tf
import resampy
from pydub import AudioSegment
from os import path
# import tensorflow.keras as keras

app = Flask(__name__)
model = tf.keras.models.load_model("./audio_classifier/ann.hdf5")

@app.route("/", methods = ['GET','POST'])
@app.route("/home", methods = ['GET','POST'])
def home() :
    return render_template('index.html')

@app.route("/submit", methods = ['POST'])
def submit():
    if 'myfile' not in request.files:
        return render_template('index.html',message='No file uploaded'),400
    myfile = request.files['myfile']
    if myfile.filename == "":
        return render_template('index.html',message='No file uploaded'),400
        
    def features_extractor(filename):
        audio , sample_rate = librosa.load(filename,res_type='kaiser_fast')
        mfccs_features = librosa.feature.mfcc(y=audio,sr=sample_rate,n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        return mfccs_scaled_features
    
    prediction_features = features_extractor(myfile)
    prediction_features = prediction_features.reshape(1,-1)
    predict=model.predict(prediction_features) 
    predict_class=np.argmax(predict,axis=1)
    if(predict_class == 0):
        return render_template('index.html',message='File is uploaded Successfully!', prediction_text="Genre is blues")
    elif predict_class==1:
        return render_template('index.html',message='File is uploaded Successfully!', prediction_text="Genre is classical")
    elif predict_class==2:
        return render_template('index.html',message='File is uploaded Successfully!', prediction_text="Genre is country")
    elif predict_class==3:
        return render_template('index.html',message='File is uploaded Successfully!', prediction_text="Genre is disco")
    elif predict_class==4:
        return render_template('index.html',message='File is uploaded Successfully!', prediction_text="Genre is hiphop")
    elif predict_class==5:
        return render_template('index.html',message='File is uploaded Successfully!', prediction_text="Genre is jazz")
    elif predict_class==6:
        return render_template('index.html',message='File is uploaded Successfully!', prediction_text="Genre is metal")
    elif predict_class==7:
        return render_template('index.html',message='File is uploaded Successfully!', prediction_text="Genre is pop")
    elif predict_class==8:
        return render_template('index.html',message='File is uploaded Successfully!', prediction_text="Genre is reggae")
    else:
        return render_template('index.html',message='File is uploaded Successfully!', prediction_text="Genre is rock")
    
    # return render_template('index.html',message='File is uploaded Successfully!', prediction_text="this is the prediction")

if __name__ =="__main__":
    app.run(debug=True)


