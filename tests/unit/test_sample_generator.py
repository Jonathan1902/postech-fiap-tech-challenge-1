
from churn_predictor.domain.schemas import CustomerProfile
from churn_predictor.utils.sample_generator import RandomCustomerGenerator


def test_generator_produces_valid_profiles():
    gen = RandomCustomerGenerator(seed=42)
    profiles = gen.generate_batch(1000)
    for p in profiles:
        assert isinstance(p, CustomerProfile)
        assert 0 <= p.tenure_months <= 72
        assert p.monthly_charges > 0


def test_generator_reproducible():
    g1 = RandomCustomerGenerator(seed=0)
    g2 = RandomCustomerGenerator(seed=0)
    p1 = g1.generate()
    p2 = g2.generate()
    assert p1.tenure_months == p2.tenure_months
    assert p1.monthly_charges == p2.monthly_charges


def test_tenure_zero_has_zero_total_charges():
    gen = RandomCustomerGenerator(seed=0)
    for _ in range(200):
        p = gen.generate()
        if p.tenure_months == 0:
            assert p.total_charges == 0.0
            return
