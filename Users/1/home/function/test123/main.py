import os, time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# API 
app = FastAPI(root_path='/', docs_url=None, redoc_url=None)
# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

index = 0

@app.get("/test")
def test():
    global index
    index += 1
    print(f'A: {index}')
    time.sleep(3)
    print(f'B: {index}')
    index -= 1
    return {"msg": "ok"}

# import uvicorn
# uvicorn.run(app, host="0.0.0.0", port=7000)

