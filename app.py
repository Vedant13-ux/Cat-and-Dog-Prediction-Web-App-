from logging import debug
from flask import Flask, views, request, render_template, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
import os 

# Initializing the Flask Application 
app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = './'
cnn = tf.keras.models.load_model('Cat_or_Dog')


@app.route('/predict', methods=["GET","POST"])
def api():
    if(request.method=="GET"):
        return render_template('index.html')
    if(request.method=="POST"):
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        import numpy as np
        from keras.preprocessing import image
        test_image = image.load_img(secure_filename(f.filename), target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = cnn.predict(test_image)
        os.remove(secure_filename(f.filename))
        if result[0][0] == 1:
            return render_template('index.html', prediction= "Dog")
        else:
            return render_template('index.html', prediction= "Cat")
            

if __name__=="__main__":
    app.run(debug=True)


