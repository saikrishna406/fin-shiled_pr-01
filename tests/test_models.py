from finshield.data.generator import SyntheticDataGenerator
from finshield.features.extractor import FeatureExtractor
from finshield.models.trainer import ModelTrainer
from finshield.models.evaluator import ModelEvaluator

def test_model_training_and_eval():
    gen = SyntheticDataGenerator(num_transactions=200, seed=42)
    raw_df = gen.generate()
    extractor = FeatureExtractor()
    feat_df = extractor.fit_transform(raw_df)

    trainer = ModelTrainer()
    models = trainer.train_all(feat_df)

    assert "xgboost" in models
    assert "isolation_forest" in models

    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all(models, feat_df)

    assert "xgboost" in results
    assert results["xgboost"]["accuracy"] >= 0.0
    assert results["xgboost"]["f1_score"] >= 0.0
