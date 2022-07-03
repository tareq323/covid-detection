from flask import Flask, render_template, request
import PIL
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)

dic = {0 : 'COVID19', 1 : 'NORMAL',2: 'PNEUMONIA'}

model = load_model('covid_model.h5')

model.make_predict_function()

def predict_label(img_path):
	i = image.image_utils.load_img(img_path, target_size=(224,224))
	i = image.image_utils.img_to_array(i)/255.0
	i = i.reshape(1,224,224,3)
	p = model.predict(i)
	return dic[np.argmax(p.round(2))]


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Covid-19 CNN"

@app.route("/testt")
def innded():
	return render_template("testt.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("testt.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = False)