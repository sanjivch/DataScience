from fastapi import FastAPI 
app = FastAPI(debug=True)

@app.get("/")
async def root():
    return {"message": "Hello World 445"}

@app.get("/items/{name}")
async def get_items(name):
    return {"message": name}