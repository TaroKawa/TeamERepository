from flask import Flask, render_template
# GETパラメーターを取得するために、requestを追加する
from flask import request

app = Flask(__name__)
print ("test4 Post works!")

@app.route('/')
def index():
    return render_template('index.html', message="こんにちは")
@app.route("/page1")
def page1():
    return render_template('page1.html')

@app.route('/select_color')
def select_color():
    return render_template('select_color.html')

@app.route('/create_new_clothes', methods=['POST'])
# methodsにPOSTを指定すると、POSTリクエストを受けられる
def create_new_clothes():

    # request.formにPOSTデータがある
    red = request.form["red"]
    green = request.form["green"]
    blue = request.form["blue"]
    print("your color selection is:")
    print("Red;"+str(red)+" Green;"+str(green)+" Blue;"+str(blue))

    return render_template('select_color.html', red_from_post=red,green_from_post=green,blue_from_post=blue)


if __name__ == "__main__":
    app.run(debug=True)
