from enum import StrEnum


class YesNo(StrEnum):
    yes = "Yes"
    no = "No"


class InternetService(StrEnum):
    dsl = "DSL"
    fiber_optic = "Fiber optic"
    cable = "Cable"
    no = "No"


class ContractType(StrEnum):
    month_to_month = "Month-to-month"
    one_year = "One year"
    two_year = "Two year"


class PaymentMethod(StrEnum):
    electronic_check = "Electronic check"
    mailed_check = "Mailed check"
    bank_transfer = "Bank transfer (automatic)"
    credit_card = "Credit card (automatic)"
