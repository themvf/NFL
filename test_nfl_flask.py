import sys
sys.path.append('src')

try:
    print("Testing NFL Flask app imports...")
    from nfl_app.app.flask_app import app
    print("✓ Flask app imported successfully")
    
    print("Testing route...")
    with app.test_client() as client:
        resp = client.get('/')
        print(f"✓ Route test: {resp.status_code}")
        
    print("Starting the app...")
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
