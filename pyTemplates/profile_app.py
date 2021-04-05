from flask import Flask, request, render_template
import pandas as pd
profile_app = Flask(__name__)


@profile_app.route("/")
def main():
    return 'Welcome'


@profile_app.route("/home", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # check if the post request has the file part
        file = request.files['file']
    return render_template('home.html')


if __name__ == '__main__':
    profile_app.run(debug=True)

