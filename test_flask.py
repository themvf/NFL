from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return '<h1>Hello from Flask!</h1><p>This is a test to see if Flask works.</p>'

if __name__ == '__main__':
    print("Starting Flask test app...")
    print("If successful, open: http://127.0.0.1:5000")
    try:
        app.run(host='127.0.0.1', port=5000, debug=True)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
