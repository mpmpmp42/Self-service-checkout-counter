from flask import Flask,render_template

app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('实验.html')

@app.route('/')
def ceshi():
    # return render_template('ceshi.html')
    fanhui = [{
  "name": "apple",
  "price": 3,
  "number": 2,
  "weight": 500
 },
 {
  "name": "banana",
  "price": 5,
  "number": 6,
  "weight": 250
 },
 {
  "name": "coffee",
  "price": 15,
  "number": 3,
  "weight": 1000
 }
 ]
    return render_template('newUI.html',fanhui=fanhui)

if __name__ == '__main__':
    app.debug=True
    app.run()
