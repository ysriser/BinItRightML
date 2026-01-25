from typing import Union

from fastapi import FastAPI
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    # Fix for Alert 10021
    response.headers["X-Content-Type-Options"] = "nosniff"
    # Fix for Alert 90004
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    return response