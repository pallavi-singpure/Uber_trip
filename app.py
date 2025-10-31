from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import Base, Prediction, predict_trips, dispatching_options

# Initialize FastAPI app
app = FastAPI()

# Create tables
Base.metadata.create_all(bind=engine)

# Mount folders
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/", response_class=HTMLResponse)
def home(request: Request, db: Session = Depends(get_db)):
    recent_predictions = db.query(Prediction).order_by(Prediction.id.desc()).limit(5).all()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "recent_predictions": recent_predictions, "dispatching_options": dispatching_options, "result": None}
    )

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    dispatching_base_number: str = Form(...),
    active_vehicles: float = Form(...),
    day: int = Form(...),
    month: int = Form(...),
    year: int = Form(...),
    db: Session = Depends(get_db)
):
    # Make prediction
    prediction = predict_trips(dispatching_base_number, active_vehicles, day, month, year)

    # Store record in DB
    new_record = Prediction(
        dispatching_base_number=dispatching_base_number,
        active_vehicles=active_vehicles,
        day=day,
        month=month,
        year=year,
        prediction=prediction
    )
    db.add(new_record)
    db.commit()
    db.refresh(new_record)

    # Fetch recent predictions
    recent_predictions = db.query(Prediction).order_by(Prediction.id.desc()).limit(5).all()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": prediction,
            "dispatching_options": dispatching_options,
            "recent_predictions": recent_predictions
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=5050, reload=True)
