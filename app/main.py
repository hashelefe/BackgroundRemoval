from fastapi import FastAPI, Request
from app.model import api_call

app = FastAPI(title="Remove Background", description="Remove Background API", version="1.0.0")

@app.get("/")
def home():
    return {"message": "Hello World"}
    
@app.post("/remove-background", response_model=str, status_code=200, tags=["Remove Background"])
def remove_background(data:dict):
    return api_call(str(data['input_url']))

@app.post("/data/")
async def get_data(data: dict):
    return {"message": "This is data:", "data": data}