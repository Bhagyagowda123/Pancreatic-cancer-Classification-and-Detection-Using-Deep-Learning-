from flask import Flask, render_template, url_for, request
import sqlite3
import os
import numpy as np
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import shutil
import cv2




# Set parameters
IMG_HEIGHT, IMG_WIDTH = 150, 150  # Adjust based on your dataset
MODEL_PATH = 'pancreatic_tumor_model.h5'


# Load the trained model
model = load_model(MODEL_PATH)

# Load and preprocess the image
def load_and_preprocess_image(img_path):
    img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))  # Load image
    img_array = img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Rescale pixel values
    return img_array

connection = sqlite3.connect('user_data.db')
cursor = connection.cursor()

command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
cursor.execute(command)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result)==0:
            return render_template('index.html',msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return render_template('userlog.html')

    return render_template('home.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')



@app.route('/userlog.html', methods=['GET'])
def indexBt():
      return render_template('userlog.html')


@app.route('/graph.html', methods=['GET', 'POST'])
def graph():
    
    images = ['http://127.0.0.1:5000/static/accuracy_loss.png',
              
              'http://127.0.0.1:5000/static/confusion_matrix.png']
    content=['Accuracy Graph',
            'Confusion Matrix']

            
    
        
    return render_template('graph.html',images=images,content=content)
    



@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
        
        dirPath = "static/images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        fileName=request.form['filename']
        dst = "static/images"
        
        shutil.copy("test/"+fileName, dst)
        image = cv2.imread("test/"+fileName)
        # #color conversion
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('static/gray.jpg', gray_image)
        # #apply the Canny edge detection
        edges = cv2.Canny(image, 250, 254)
        cv2.imwrite('static/edges.jpg', edges)
        # #apply thresholding to segment the image
        retval2,threshold2 = cv2.threshold(gray_image,128,255,cv2.THRESH_BINARY)
        cv2.imwrite('static/threshold.jpg', threshold2)
        # # create the sharpening kernel
        kernel_sharpening = np.array([[-1,-1,-1],
                                     [-1, 9,-1],
                                    [-1,-1,-1]])

        # # apply the sharpening kernel to the image
        sharpened =cv2.filter2D(image, -1, kernel_sharpening)

        # save the sharpened image
        cv2.imwrite('static/sharpened.jpg', sharpened)

        # Preprocess the single image
        input_image = load_and_preprocess_image("test/"+fileName)
        predictions = model.predict(input_image)

        predicted_class = np.argmax(predictions, axis=1)
        
        print("Predicted class:", predicted_class)


        
       
        str_label=""
        accuracy=0.0
        Med=""
        Med1=""
        food=""
        food1=""
        if predicted_class[0] == 0:
            str_label="Acute pancreatitis(stage1)"
            accuracy = (predicted_class[0] == 0).astype(float)
            print(f'Accuracy: {accuracy * 100:.2f}%')
            Med="The Medical Treatment"
            Med1=["Acute pancreatitis is typically treated with supportive care.",
            "Provide IV fluids to prevent dehydration.",
            "Use pain relief medication to manage abdominal pain"]
            food="Food Recommendation"
            food1=[" A low-fat, easily digestible diet aids recovery.",
            "Consume clear liquids like broths and gelatin.",
            "Gradually reintroduce low-fat foods as inflammation reduces."]


        elif predicted_class[0] == 1:
            str_label="Adenosquamous carcinoma(stage2)"
            accuracy = (predicted_class[0] == 1).astype(float)
            print(f'Accuracy: {accuracy * 100:.2f}%')
            Med="The Medical Treatment"
            Med1=["Adenosquamous carcinoma is treated aggressively with a multimodal approach.",
            "Surgery is typically followed by radiation to remove the tumor.",
            "Chemotherapy may be added for advanced cases."]
            food="Food Recommendation"
            food1=[" A balanced, soft diet supports recovery post-treatment.",
            "Lean proteins like chicken and fish help rebuild strength.",
            "Soft, easily digestible vegetables like steamed carrots and spinach are recommended."]


        elif predicted_class[0] == 2:
            str_label="Chronic pancreatitis(stage3)"
            accuracy = (predicted_class[0] == 2).astype(float)
            print(f'Accuracy: {accuracy * 100:.2f}%')
            Med="The Medical Treatment"
            Med1=["Chronic pancreatitis is managed through long-term lifestyle changes.",
            "Prescribe enzyme supplements to aid digestion.",
            "Administer analgesics for ongoing pain management."]
            food="Food Recommendation"
            food1=[" Dietary changes are crucial to avoid symptom flare-ups.",
            "Eat small, frequent meals that are low in fat.",
            "Incorporate high-fiber fruits and vegetables to ease digestion"]

        elif predicted_class[0] ==3:
            str_label="Metastatic pancreatic cancer(final stage)"
            accuracy = (predicted_class[0] == 3).astype(float)
            print(f'Accuracy: {accuracy * 100:.2f}%')
            Med="The Medical Treatment"
            Med1=["Metastatic pancreatic cancer requires systemic therapy.",
            "Chemotherapy is often used to slow the spread of cancer.",
            "Targeted therapies can focus on specific cancer cell markers."]
            food="Food Recommendation"
            food1=["Nutrition is important to maintain strength during treatment.",
            "Opt for calorie-rich, high-protein foods to combat weight loss."
            "Include smoothies and shakes for easy digestion and nutrient intake."]

        elif predicted_class[0] == 4:
            str_label="Normal"
            accuracy = (predicted_class[0] == 4).astype(float)
            print(f'Accuracy: {accuracy * 100:.2f}%')


        
            

       



        
        
        
        return render_template('results.html', status=str_label,status2=f'accuracy is {accuracy}',Treatment=Med,Treatment1=Med1,FoodRecommendation=food,FoodRecommendation1=food1,ImageDisplay="http://127.0.0.1:5000/static/images/"+fileName,ImageDisplay1="http://127.0.0.1:5000/static/gray.jpg",ImageDisplay2="http://127.0.0.1:5000/static/edges.jpg",ImageDisplay3="http://127.0.0.1:5000/static/threshold.jpg",ImageDisplay4="http://127.0.0.1:5000/static/sharpened.jpg")
    return render_template('index.html')




@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
