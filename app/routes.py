import asyncio
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, WebSocket
from fastapi.responses import JSONResponse
from . import schemas as schemas
from .import service as serv 

router = APIRouter()

@router.post('/upload')
async def upload_video(file: UploadFile = File(...)):
    try:
        result = await serv.upload_video(file)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при загрузке видео: {str(e)}")


@router.websocket('/ws/{video_id}')
async def websocket_endpoint(websocket: WebSocket, video_id: str):
    await websocket.accept()
    await websocket.send_text('ээ бля')

    try:
        for res in serv.process_file(video_id):
            await websocket.send_json(res.model_dump_json())

            # await asyncio.sleep(1)  # Пауза 1 секунда
            
    except Exception as e:
        await websocket.send({
            "status": "error",
            "description": f"{e}"
        })
    finally:
        await websocket.close()