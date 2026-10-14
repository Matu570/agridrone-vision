import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from database import Base, engine, get_db
from models import Scan
from vision import InvalidImageError, detect_and_count_crops

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
PROCESSED_DIR = BASE_DIR / "static" / "processed"
TEMPLATE_PATH = BASE_DIR / "templates" / "index.html"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="AgriDrone Vision", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def home():
    if not TEMPLATE_PATH.exists():
        raise HTTPException(status_code=500, detail="No se encontró la interfaz.")
    return FileResponse(TEMPLATE_PATH)


@app.get("/api/scans")
def list_scans(db: Session = Depends(get_db)):
    scans = db.query(Scan).order_by(Scan.created_at.desc()).all()
    return [
        {
            "id": scan.id,
            "filename": scan.filename,
            "created_at": scan.created_at.replace(tzinfo=timezone.utc).isoformat(),
            "total_count": scan.total_count,
            "original_url": f"/static/uploads/{scan.filename}",
            "processed_url": f"/static/processed/{scan.filename}",
        }
        for scan in scans
    ]


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    extension = Path(file.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Formato no permitido. Sube una imagen JPG o PNG.",
        )

    safe_name = f"{uuid.uuid4().hex}{extension}"
    original_path = UPLOAD_DIR / safe_name
    processed_path = PROCESSED_DIR / safe_name

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")

    try:
        original_path.write_bytes(contents)
        _, total_count = detect_and_count_crops(str(original_path), str(processed_path))
    except InvalidImageError as exc:
        if original_path.exists():
            original_path.unlink()
        if processed_path.exists():
            processed_path.unlink()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        if original_path.exists():
            original_path.unlink()
        if processed_path.exists():
            processed_path.unlink()
        raise HTTPException(
            status_code=500,
            detail="Ocurrió un error al procesar la imagen.",
        )

    scan = Scan(
        filename=safe_name,
        created_at=datetime.utcnow(),
        total_count=total_count,
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)

    return {
        "id": scan.id,
        "filename": scan.filename,
        "created_at": scan.created_at.replace(tzinfo=timezone.utc).isoformat(),
        "total_count": scan.total_count,
        "original_url": f"/static/uploads/{scan.filename}",
        "processed_url": f"/static/processed/{scan.filename}",
    }


@app.get("/health")
def health():
    return {"status": "ok"}
