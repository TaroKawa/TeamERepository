import os
from flask import Flask, render_template

#相対パスでテンプレート指定
app = Flask(__name__, template_folder='app/templates')


# app.py
@app.route('/')
def index():
    # 「templates/index.html」のテンプレートを使う
    # 「message」という変数に"Hello"と代入した状態で、テンプレート内で使う
    return render_template('index.html', message="Hello World!")



if __name__ == "__main__":
    app.run('127.0.0.1', 5000, debug=True)
