from flask import Flask, render_template, request, flash, redirect
from werkzeug import secure_filename
import os
import churn_model

FILE_NAME = ''

app = Flask(__name__)
app.secret_key = "secret key"



@app.route("/")
def home():
    return render_template("home.html")
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		f = request.files['file']
		f.save(secure_filename(f.filename)) 
		
		
		cm, accuracy = churn_model.train_model(f.filename)
		flash('Confusion Matrix')
		flash('cm')
		flash('Accuracy')
		flash(accuracy)

		
		
		return redirect('/')
		
		
 
@app.route('/model', methods = ['GET', 'POST']) 
def model():
	flash(FILE_NAME)
	return redirect('/')
  	  
	  
	  

	
if __name__ == "__main__":
    app.run(debug=True)