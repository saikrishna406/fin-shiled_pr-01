from finshield.data.generator import SyntheticDataGenerator
from finshield.features.extractor import FeatureExtractor, get_feature_columns

def test_feature_extractor():
    gen = SyntheticDataGenerator(num_transactions=50, seed=42)
    raw_df = gen.generate()
    extractor = FeatureExtractor()
    feat_df = extractor.fit_transform(raw_df)

    feature_cols = get_feature_columns()
    for col in feature_cols:
        assert col in feat_df.columns
    assert len(feat_df) == 50
