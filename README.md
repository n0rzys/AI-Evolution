# AI Evolution

Проект демонстрирует эволюцию интеллектуальных систем: от экспертных правил к современным моделям машинного обучения.

## Идея

Раньше системы принимали решения по заранее заданным правилам (экспертные системы).  
Сейчас модели обучаются на данных и сами находят закономерности.

В проекте это показано на задаче классификации риска сердечных заболеваний.

---

## Что реализовано

- XGBoost модель  
- MLP (PyTorch)  
- FastAPI API  
- Swagger UI  
- Docker + Docker Compose  

---

## Запуск

docker compose up --build

После запуска:

http://127.0.0.1:8000
http://127.0.0.1:8000/docs

Пример запроса:
{
  "BMI": 27.5,
  "Smoking": "No",
  "AlcoholDrinking": "No",
  "Stroke": "No",
  "PhysicalHealth": 2,
  "MentalHealth": 1,
  "DiffWalking": "No",
  "Sex": "Male",
  "AgeCategory": "55-59",
  "Race": "White",
  "Diabetic": "No",
  "PhysicalActivity": "Yes",
  "GenHealth": "Good",
  "SleepTime": 7,
  "Asthma": "No",
  "KidneyDisease": "No",
  "SkinCancer": "No"
}

Архитектура:
FastAPI → preprocessing → ML модель → ответ

Технологии:
Python, FastAPI, PyTorch, XGBoost, Docker
