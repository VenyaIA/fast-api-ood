import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from io import BytesIO
from PIL import Image
from model_ood import detect_objects

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Model server is running"}


@app.post("/upload")
async def uploads_multiple_files(upload_file: UploadFile = File(...)):
    file_content = await upload_file.read()
    file_like = BytesIO(file_content)

    angle_degrees, annotated_img = detect_objects(file_like)
    if angle_degrees:
        angle_degrees = ", ".join(angle_degrees)
    else:
        angle_degrees = 'No oriented bounding boxes detected.'

    # Преобразование аннотированного изображения в байты
    img_byte_arr = BytesIO()
    annotated_img_pil = Image.fromarray(annotated_img)  # Преобразуем numpy-массив в PIL изображение
    annotated_img_pil.save(img_byte_arr, format="PNG")  # Сохраняем как PNG
    img_byte_arr.seek(0)  # Возвращаемся к началу потока

    return StreamingResponse(
        img_byte_arr,
        media_type="image/png",
        headers={
            "Content-Disposition": "attachment; filename=annotated_image.png",
            "Angles-Degrees": angle_degrees  # Передаем углы через заголовки
        },
    )


if __name__ == "__main__":
    uvicorn.run(app="main:app")

