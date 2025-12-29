import pytest
from signwriting_animation.data.data_loader import DynamicPosePredictionDataset


@pytest.fixture(scope="module")
def pose_dataset():
    """
    Fixture that provides a DynamicPosePredictionDataset instance
    for use in test modules within this package.
    """
    num_past_frames_max = 8
    num_future_frames_max = 5
    num_keypoints = 586
    num_dims_per_keypoint = 3

    dataset = DynamicPosePredictionDataset(
        data_dir='signwriting_animation/data/test_data/pose_samples',
        csv_path='signwriting_animation/data/test_data/transcription_samples/data_subset.csv',
        num_past_frames=num_past_frames_max,
        num_future_frames=num_future_frames_max,
        with_metadata=False,
        split='train'
    )

    return dataset, num_past_frames_max, num_future_frames_max, num_keypoints, num_dims_per_keypoint
