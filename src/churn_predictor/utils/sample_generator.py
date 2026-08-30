from __future__ import annotations

import random

from churn_predictor.domain.enums import (
    ContractType,
    InternetService,
    PaymentMethod,
    SeniorCitizen,
    YesNo,
)
from churn_predictor.domain.schemas import CustomerProfile


class RandomCustomerGenerator:
    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def generate(self) -> CustomerProfile:
        tenure = self._rng.randint(0, 72)
        monthly = round(self._rng.uniform(18.25, 118.75), 2)
        if tenure == 0:
            total = 0.0
        else:
            jitter = self._rng.uniform(0.9, 1.1)
            total = round(tenure * monthly * jitter, 2)

        return CustomerProfile.model_validate(
            {
                "Tenure Months": tenure,
                "Monthly Charges": monthly,
                "Total Charges": total,
                "Senior Citizen": self._rng.choice(list(SeniorCitizen)).value,
                "Partner": self._rng.choice(list(YesNo)).value,
                "Dependents": self._rng.choice(list(YesNo)).value,
                "Multiple Lines": self._rng.choice(list(YesNo)).value,
                "Internet Service": self._rng.choice(list(InternetService)).value,
                "Online Security": self._rng.choice(list(YesNo)).value,
                "Online Backup": self._rng.choice(list(YesNo)).value,
                "Device Protection": self._rng.choice(list(YesNo)).value,
                "Tech Support": self._rng.choice(list(YesNo)).value,
                "Streaming TV": self._rng.choice(list(YesNo)).value,
                "Streaming Movies": self._rng.choice(list(YesNo)).value,
                "Contract": self._rng.choice(list(ContractType)).value,
                "Paperless Billing": self._rng.choice(list(YesNo)).value,
                "Payment Method": self._rng.choice(list(PaymentMethod)).value,
            }
        )

    def generate_batch(self, n: int) -> list[CustomerProfile]:
        return [self.generate() for _ in range(n)]
