"""Quick image check: upload one picture and get a verdict without an area, time or AIS file."""
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.auth import get_current_user
from app.models import User

router = APIRouter(prefix="/api", tags=["quick check"])
MAX_BYTES = 25 * 1024 * 1024


@router.post("/quick-check")
def quick_check(file: UploadFile = File(...), _: User = Depends(get_current_user)):
    data = file.file.read(MAX_BYTES + 1)
    if not data:
        raise HTTPException(400, "The file is empty.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "The image is larger than 25 MB.")
    from app.services.analysis_service import get_detector
    from ml.quickcheck import check
    try:
        result = check(data, detector=get_detector())
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    result["filename"] = file.filename
    return result
