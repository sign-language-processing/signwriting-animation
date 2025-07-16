import pytest
from signwriting_animation.data.data_loader import DatasetConfig, DynamicPosePredictionDataset

@pytest.fixture(scope="module")
def pose_dataset():
    """
    Fixture that provides a DynamicPosePredictionDataset instance
    for use in test modules within this package.
    """
    config = DatasetConfig(
        data_dir='signwriting_animation/data/test_data/pose_samples',
        csv_path='signwriting_animation/data/test_data/transcription_samples/data_subset.csv',
        num_past_frames=8,
        num_future_frames=5,
        split='train'
    )
    dataset = DynamicPosePredictionDataset(
        config=config,
        with_metadata=False
    )
    return dataset
