from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def root():
    return {
        "message": "Sign Language Translator API is running"
    }