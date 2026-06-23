from pydantic import BaseModel, ConfigDict, Field


class CreditApplication(BaseModel):
    LIMIT_BAL: float = Field(..., gt=0, description="Кредитный лимит клиента (NT$)")
    EDUCATION: int = Field(..., ge=1, le=4, description="1=graduate school, 2=university, 3=high school, 4=others")
    MARRIAGE: int = Field(..., ge=1, le=3, description="1=married, 2=single, 3=others")
    AGE: int = Field(..., ge=18, le=100)
    max_delay: float = Field(..., description="Максимальный код задержки платежа за 6 месяцев")
    mean_delay: float = Field(..., description="Средний код задержки платежа за 6 месяцев")
    n_months_delayed: int = Field(..., ge=0, le=6, description="Число месяцев из 6 с задержкой платежа")
    avg_bill_amt: float = Field(..., description="Средний размер счёта за 6 месяцев")
    avg_pay_amt: float = Field(..., ge=0, description="Средний размер платежа за 6 месяцев")
    payment_to_bill_ratio: float = Field(..., description="Отношение суммарных платежей к суммарным счетам")
    credit_utilization: float = Field(..., description="Средний счёт / кредитный лимит")
    bill_trend: float = Field(..., description="Тренд изменения суммы счёта за 6 месяцев")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "LIMIT_BAL": 200000,
            "EDUCATION": 2,
            "MARRIAGE": 1,
            "AGE": 35,
            "max_delay": 0,
            "mean_delay": -0.5,
            "n_months_delayed": 0,
            "avg_bill_amt": 45000,
            "avg_pay_amt": 6000,
            "payment_to_bill_ratio": 0.8,
            "credit_utilization": 0.225,
            "bill_trend": 1200.0,
        }
    })


class PredictionResponse(BaseModel):
    prediction: int
    proba_default: float
    risk_category: str
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
