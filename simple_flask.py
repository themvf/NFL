from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return '''
    <h1>Simple NFL Data Viewer</h1>
    <p>This is a basic Flask app to test if the framework works.</p>
    <p>If you see this, Flask is working correctly!</p>
    '''

if __name__ == '__main__':
    print("=" * 50)
    print("STARTING FLASK APP")
    print("Open browser to: http://127.0.0.1:5000")
    print("=" * 50)
    
    # Run without debug mode and with explicit threading
    app.run(
        host='127.0.0.1', 
        port=5000, 
        debug=False,
        threaded=True,
        use_reloader=False
    )
