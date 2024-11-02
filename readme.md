# How to run the app
    'python -m uvicorn app:app --reload'
    'curl -X POST "http://127.0.0.1:8000/upload/" -F "files=@filepath\x.pdf" -F "files=@filepathy.pdf" '
    'curl -X POST "http://127.0.0.1:8000/answer/" -H "Content-Type: application/json" -d "{\"question\": \"What is the main topic of the document?\", \"k\": 3}" '

    

