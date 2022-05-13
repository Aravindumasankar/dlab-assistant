# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START gae_flex_quickstart]
import os
import importlib
from pymongo import MongoClient
import utils
import logging
import uuid
import face.recognition as face_recognition
import face.emotion as emotion
import ocr
from datetime import date, datetime
from functools import wraps
from flask import Flask, request, jsonify, abort, send_from_directory, send_file
import json
import wget
import object_detection

logging.basicConfig(filename='app.log')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

with open('config.json') as f:
    app_config = json.load(f)

TODAY_MODEL = str(date.today())
API_KEY = app_config['app']['API_KEY']
try:
    conn = MongoClient('mongodb://localhost:27017/')
    db = conn.memes
    collection = db.meme
    print("Connected successfully!!!")
except:
    print("Could not connect to MongoDB")

app = Flask(__name__)


# The actual decorator function check api key
def require_appkey(view_function):
    @wraps(view_function)
    # the new, post-decoration function. Note *args and **kwargs here.
    def decorated_function(*args, **kwargs):
        if request.headers.get('x-api-key') and request.headers.get('x-api-key') == API_KEY:
            return view_function(*args, **kwargs)
        else:
            abort(401)

    return decorated_function


@app.route('/')
def info():
    data = {'available_modules': app_config['modules']}
    resp = jsonify(data)
    resp.headers.add('Access-Control-Allow-Origin', '*')
    resp.status_code = 200
    return resp


@app.route('/<module_name>/info')
def module_info(module_name):
    if module_name in app_config['modules']:
        data = {'module_info': app_config['modules'][module_name]}
        resp = jsonify(data)
        resp.status_code = 200
        return resp
    else:
        abort(404)


@app.route('/<module_name>/<module_type>/info')
def module_intro(module_name, module_type):
    if (module_name in app_config['modules']) and (module_type in app_config['modules'][module_name]):
        data = app_config['modules'][module_name][module_type]
        resp = jsonify(data)
        resp.status_code = 200
        return resp
    else:
        abort(404)


@app.route('/<module_name>/<module_type>/models/')
@require_appkey
def get_models(module_name, module_type):
    print(module_name)
    if (module_name in app_config['modules']) and (module_type in app_config['modules'][module_name]):
        module_config = app_config['modules'][module_name][module_type]
        data = {}
        count = 0
        data['current_model'] = []
        data['base_model'] = []
        data['available_models'] = []
        model_file_dates = []
        for model_file in os.listdir(module_name + '/' + module_type + '/' + module_config['MODEL_DIR']):
            full_file_path = os.path.join(module_name + '/' + module_type + '/' + module_config['MODEL_DIR'],
                                          model_file)
            if model_file.startswith('base_'):
                data['base_model'].append(
                    {'name': request.url_root + full_file_path,
                     'filename': model_file,
                     'size': utils.file_size(full_file_path)})
            else:
                model_file_dates.append(datetime.strptime(model_file.split('_')[0], '%Y-%m-%d').date())

            data['available_models'].append(
                {'name': request.url_root + full_file_path,
                 'filename': model_file,
                 'size': utils.file_size(full_file_path)})

            count += 1
        for model_file in os.listdir(module_name + '/' + module_type + '/' + module_config['MODEL_DIR']):
            full_file_path = os.path.join(module_name + '/' + module_type + '/' + module_config['MODEL_DIR'],
                                          model_file)
            if model_file.startswith(str(max(model_file_dates))):
                data['current_model'].append(
                    {'name': request.url_root + full_file_path,
                     'filename': model_file,
                     'size': utils.file_size(full_file_path)})
        data['count'] = count
        resp = jsonify(data)
        resp.status_code = 200
        return resp
    else:
        abort(404)


@app.route('/<module_name>/<module_type>/model/<path:filename>')
@require_appkey
def download_model_file(module_name, module_type, filename):
    if (module_name in app_config['modules']) and (module_type in app_config['modules'][module_name]):
        module_config = app_config['modules'][module_name][module_type]
        return send_from_directory('../' + module_name + '/' + module_type + '/' + module_config['MODEL_DIR'],
                                   filename, as_attachment=True)
    else:
        abort(404)


@app.route('/<module_name>/<module_type>/model/train', methods=['POST'])
@require_appkey
def train(module_name, module_type):
    if (module_name in app_config['modules']) and (module_type in app_config['modules'][module_name]):
        data = {}
        params = request.get_json(force=True)
        dynamic_module = importlib.import_module(module_name + '.' + module_type)
        module_type = app_config['modules'][module_name][module_type]
        today_model = str(date.today()) + '_trained_' + module_type['NAME'] + '_model' + module_type['MODEL_EXTENSION']
        train_dir = module_type['MODULE_URL'] + module_type['TRAIN_DIR']
        data['module_type'] = module_type
        data['model_file'] = module_type['MODULE_URL'] + module_type['MODEL_DIR'] + today_model
        data['model_file_link'] = request.url_root + data['model_file']
        data['msg'] = "model trained successfully."

        def train_module(module_type_name, params):
            switcher = {
                'recognition':
                    dynamic_module.train(train_dir, data['model_file'], params['n_neighbours']),
            }
            return switcher.get(module_type_name, params)

        # if train_module(module_type['NAME'],params) != -1:
        #     resp = jsonify(data)
        #     resp.status_code = 200
        # else:
        train_module(module_type['NAME'], params)

        data['msg'] = "model trained."
        resp = jsonify(data)
        resp.status_code = 200
        return resp

    else:
        abort(404)


@app.route('/<dir_name>/<path:filename>')
def download_file(dir_name, filename):
    file_path = '../' + dir_name + '/' + filename
    print(file_path)
    return send_file(file_path)


@app.route('/meme/classify', methods=['POST'])
@require_appkey
def classify():
    upload_dir = 'uploads/'
    # Create directory 'updated_images' if it does not exist
    if not os.path.exists(upload_dir):
        print("Uploads directory created")
        os.makedirs(upload_dir)
    data = {}
    # check if the post request has the file part
    if 'file' not in request.files:
        data['msg'] = 'File Required.'
        resp = jsonify(data)
        resp.status_code = 401
        return resp
    else:
        file = request.files['file']
        file_name, file_extension = os.path.splitext(file.filename)
        if file_extension in ['.png', '.jpg', '.jpeg']:
            data['original_file'] = file.filename
            data['_id'] = str(uuid.uuid4())
            file_name = data['_id'] + file_extension
            # save to UPLOAD DIR
            file.save(os.path.join(upload_dir, file_name))
            data['file_name'] = file_name
            data['file_extension'] = file_extension
            data['file_path'] = str(upload_dir + data['file_name'])
            # data['file_url'] = request.url_root + data['file_path']
            # FACE RECOGNITION
            # today_model_file = 'face/recognition/model/vadivelu_trained_knn_model.clf'
            today_model_file = 'face/recognition/model/' + str(date.today()) + '_trained_knn_model.clf'
            data['face_recogniton'] = face_recognition.predict(data['file_name'], data['file_path'], None,
                                                               today_model_file)
            # OCR
            data['ocr'] = []
            data['ocr'].append({"tresh": ocr.ocr(data['file_name'], data['file_path'], 'tresh'),
                                "blur": ocr.ocr(data['file_name'], data['file_path'], 'blur')})
            data['post_id'] = collection.insert_one(data).inserted_id

            return jsonify(data)
        else:
            data['msg'] = 'Unsupported Format. Required .png, .jpg, .jpeg'
            resp = jsonify(data)
            resp.status_code = 401
            return resp


@app.route('/meme/recognize',methods=['POST'])
def extract():
    data = {}
    image_url = request.form['image_url']
    recog = request.form['type']
    upload_dir = 'uploads/'
    # Create directory 'uploads' if it does not exist
    if not os.path.exists(upload_dir):
        print("Uploads directory created")
        os.makedirs(upload_dir)
    file_name = wget.detect_filename(image_url)
    print(file_name)
    file_path = upload_dir+file_name
    if os.path.exists(file_path):
        os.remove(file_path)
    wget.download(image_url,upload_dir)
    data['_id'] = str(uuid.uuid4())
    data['file_name'] = file_name
    data['file_path'] = file_path
    data['processed_filepath'] = 'processed/'+file_name
    if recog == 'objects' or recog == 'all':
        # OBJECT DETECTION
        data['object_detection'] = []
        data['object_detection'].append({"yolo":object_detection.detectObj(file_name,data['file_path'])})
    if recog == 'ocr' or recog == 'all':
        # OCR
        # OCR API
        payload1 = {'apikey': 'a4db52c09b88957',
                    'url': image_url,
                    'isOverlayRequired': 'True',
                    'OCREngine': '1'}
        payload2 = {'apikey': 'a4db52c09b88957',
                    'url': image_url,
                    'isOverlayRequired': 'True',
                    'OCREngine': '2'}
        data['ocr'] = []
        data['ocr'].append(
            {
                "tresh": ocr.ocr(data['file_name'], data['file_path'], 'tresh'),
                "blur": ocr.ocr(data['file_name'], data['file_path'], 'blur'),
                # "engine1": json.loads(ocr.ocr_api(image_url, payload1).text),
                # "engine2": json.loads(ocr.ocr_api(image_url, payload2).text)
                            })
    if recog == 'faces' or recog == 'all':
        # FACE RECOGNITION
        today_model_file = 'face/recognition/model/vadivelu_trained_knn_model_.clf'
        # today_model_file = 'face/recognition/model/' + str(date.today()) + '_trained_knn_model.clf'
        data['face_recogniton'] = face_recognition.predict(data['file_name'], data['file_path'], None,
                                                           today_model_file)
    response = jsonify(data)
    response.headers.add('Access-Control-Allow-Origin', '*')
    # update 'data' if 'name' exists otherwise insert new document
    # data['post_id'] = collection.find_one_and_update({"name": file_name},
    #                                {"$set": {"data": data}},
    #                                upsert=True)
    return response

@app.route('/meme/classify/')
@require_appkey
def bulk_classify():
    dataset_dir = request.args.get('dataset')
    data = {}
    today_model_file = 'face/recognition/model/' + str(date.today()) + '_trained_knn_model.clf'
    for meme in os.listdir(dataset_dir):
        print(meme)
        file_name, file_extension = os.path.splitext(meme)
        data[meme] = []
        data['original_file'] = meme
        data['_id'] = str(uuid.uuid4())
        file_name = data['_id'] + file_extension
        post = {
            "original_file": str(meme),
            "file_extension": file_extension,
            "file_path": str(dataset_dir + meme),
            "_id": data['_id'],
            "filename": file_name,
            "face_recognition": face_recognition.predict(file_name, str(dataset_dir + meme), None, today_model_file),
            "ocr": []
        }
        post['ocr'].append({"tresh": ocr.ocr(data['file_name'], data['file_path'], 'tresh'),
                            "blur": ocr.ocr(data['file_name'], data['file_path'], 'blur')})
        data[meme].append(post)
        data['post_id'] = collection.insert_one(post).inserted_id
    resp = jsonify(data)
    return resp


@app.route('/<module_name>/<module_type>/recognize', methods=['POST'])
@require_appkey
def recognize(module_name, module_type):
    if (module_name in app_config['modules']) and (module_type in app_config['modules'][module_name]):
        dynamic_module = importlib.import_module(module_name + '.' + module_type)
        module_type = app_config['modules'][module_name][module_type]
        today_model = str(date.today()) + '_trained_' + module_type['NAME'] + '_model' + module_type['MODEL_EXTENSION']
        upload_dir = module_name + '/' + module_type["NAME"] + '/' + module_type['UPLOAD_DIR']
        data = {}
        # check if the post request has the file part
        if 'file' not in request.files:
            data['msg'] = 'File Required. Formats:' + str(module_type['ALLOWED_EXTENSIONS'])
            resp = jsonify(data)
            resp.status_code = 401
        else:
            file = request.files['file']
            file_name = file.filename
            # save to UPLOAD DIR
            file.save(os.path.join(upload_dir, file_name))
            data['file_name'] = file_name
            data['file_path'] = str(upload_dir + data['file_name'])
            data['model_file'] = module_type['MODULE_URL'] + module_type['MODEL_DIR'] + today_model

            def predict_module(module_type_name):
                switcher = {
                    'recognition':
                        [
                            dynamic_module.predict(data['file_name'], data['file_path'], None, data['model_file'])

                        ]

                }
                return switcher.get(module_type_name)

            data['predictions'] = predict_module(module_type['NAME'])
            resp = jsonify(data)

        return resp
    else:
        abort(404)


@app.errorhandler(500)
def server_error(e):
    logger.exception(e)
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500
