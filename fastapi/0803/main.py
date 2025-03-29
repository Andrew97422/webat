from fastapi import FastAPI, File, UploadFile
from random import randint
from typing import List
from fastapi.responses import StreamingResponse
import io

app = FastAPI()

files_serv = []

@app.post('/api/v1/upload')
async def upload_file(
    files: List[UploadFile] = File(description="Multiple files as bytes")
):
    resp = []
    for file in files:
        content = await file.read()  # Чтение содержимого файла, если нужно
        files_serv.append(file.filename)
        resp.append(file.filename)
    return {'res': resp}

# Исправление функции для возвращения случайного файла
@app.post('/api/v1/load')
async def load_file():
    if files_serv:
        # возвращаем случайный элемент из списка файлов
        return files_serv[randint(0, len(files_serv) - 1)]
    else:
        return {"detail": "No files available"}
    
images = []
    
@app.post('/api/v1/images/upload')
async def upload_file(
    files: List[UploadFile] = File(description="Multiple files as bytes")
):
    resp = []
    for file in files:
        content = await file.read()  # Чтение содержимого файла, если нужно
        images.append({
            "content": content,
            "media_type": file.content_type  # Сохранение MIME-типа
        })
        resp.append(file.filename)
    return {'res': resp}

@app.post('/api/v1/images/load')
async def load_file():
    if images:
        # Возвращаем случайный байт-контент изображения из списка images
        selected_image = images[randint(0, len(images) - 1)]
        return StreamingResponse(io.BytesIO(selected_image["content"]), media_type=selected_image["media_type"])
    else:
        return {"detail": "No files available"}