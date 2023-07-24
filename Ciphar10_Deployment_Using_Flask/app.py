from io import BytesIO

from flask import Flask , request , render_template
import pickle
import numpy as np
from keras.preprocessing import image
import keras
import tensorflow as tf

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        in_img=request.files['image']
        in_img_process=image.load_img(BytesIO(in_img.read()),target_size=(32,32,3))
        in_image_process_array=image.img_to_array(in_img_process)
        in_image_process_array=np.expand_dims(in_image_process_array,axis=0)
        in_image_process_array /= 255
        model=tf.keras.models.load_model('Cifar10__CNN')
        result = model.predict(in_image_process_array)
        Classes= ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
        index=np.argmax(result)
        prediction=Classes[index]
        return render_template('index.html', predict=prediction)
    else:
        prediction='Please Upload an Image'
        return  render_template('index.html', predict=prediction)

if __name__ == '__main__':
    app.run(debug=True)
