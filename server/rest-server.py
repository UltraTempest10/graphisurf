#!flask/bin/python
########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
# This file implements the REST layer. It uses flask micro framework for server implementation.
# Calls from front end reaches here as json and being branched out to each project.
# Basic level of validation is also being done in this file. #
# ----------------------------------------------------------------------------------------------------------------------
########################################################################################################################
from flask import Flask, jsonify, request, redirect, render_template
from flask_httpauth import HTTPBasicAuth
from werkzeug.utils import secure_filename
import os
import shutil
import numpy as np
from search import recommend
from tensorflow.python.platform import gfile
import threading
import webbrowser

# import tarfile
# from datetime import datetime
# from scipy import ndimage
# from scipy.misc import imsave

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__, static_url_path="")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
auth = HTTPBasicAuth()


def get_result(result, classname=None):
    if result == 'static/result':
        image_path = "/result"
    else:
        image_path = os.path.join('/classified', classname)
    image_list = [os.path.join(image_path, file) for file in os.listdir(result)
                  if not file.startswith('.')]
    images = {}
    for j in range(len(image_list)):
        images['image' + str(j)] = image_list[j]
    return jsonify(images)


# ======================================================================================================================
#                                                                                                                              
#    Loading the extracted feature vectors for image retrieval                                                                 
#                                                                          						        
#                                                                                                                              
# ======================================================================================================================
num_images = 2955
extracted_features = np.zeros((num_images, 2048), dtype=np.float32)
with open('saved_features_recom.txt') as f:
    for i, line in enumerate(f):
        extracted_features[i, :] = line.split()
print("loaded extracted_features")


# ======================================================================================================================
#                                                                                                                              
#  This function is used to do the image search/image retrieval
#                                                                                                                              
# ======================================================================================================================
@app.route('/imgUpload', methods=['GET', 'POST'])
# def allowed_file(filename):
#    return '.' in filename and \
#           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def upload_img():
    print("image upload")
    result = 'static/result'
    classified = 'static/classified'
    if gfile.Exists(result):
        shutil.rmtree(result, ignore_errors=True)
    if gfile.Exists(classified):
        shutil.rmtree(classified, ignore_errors=True)

    if request.method == 'POST' or request.method == 'GET':
        print(request.method)
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)

        file = request.files['file']
        print(file.filename)
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file:  # and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            inputloc = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            recommend(inputloc, extracted_features)
            if os.path.exists(inputloc):
                os.remove(inputloc)
            return get_result(result)


# ======================================================================================================================
#
#  This function is used to get images of a particular class
#
# ======================================================================================================================
@app.route("/imgClassify")
def get_classified_img():
    print("image classify")
    classname = request.values.get('className')
    print(classname)

    if classname == 'all':
        return get_result('static/result')
    else:
        classified = os.path.join('static/classified', classname)
        return get_result(classified, classname)


# =====================================================================================================================
#                                                                                                                              
#                                           Main function                                                        	   #
#  				                                                                                                
# =====================================================================================================================
@app.route("/")
def main():
    return render_template("main.html")


if __name__ == '__main__':
    url = "http://127.0.0.1:5000"
    threading.Timer(1.25, lambda: webbrowser.open(url)).start()
    app.run(debug=True, host='0.0.0.0')
