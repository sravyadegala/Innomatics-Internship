
from flask import Flask, request, render_template,   redirect, url_for
import re

app = Flask(__name__)

matched_item_list = [] 

@app.route('/')
def home():
    return render_template("home_reg.html",matched_item_list=matched_item_list)


@app.route("/results", methods=["POST"])
def regex():
    text = request.form.get("text")
    pattern =  re.compile(request.form.get("pattern"))
    matched_items = pattern.search(text)
    if matched_items:
        extracted_string = matched_items.group()
        matched_item_list.append(extracted_string)

    return redirect(url_for('home'))


@app.route("/validateemail", methods=["POST"])
def validating_email_id():
    email = request.form.get('email')
    email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    if re.match(email_pattern, email):
        message = f"{email} is a valid email address."
    else:
        message = f"{email} is not a valid email address."
    
    return render_template("home_reg.html", email=email, message=message)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)