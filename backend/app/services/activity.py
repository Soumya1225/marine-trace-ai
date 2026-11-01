from app.models import ActivityLog


def log(db, user, action, detail=""):
    db.add(ActivityLog(user_id=user.id if user else None, action=action, detail=detail))
    db.commit()
