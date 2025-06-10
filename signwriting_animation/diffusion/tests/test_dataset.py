import pytest

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset


@pytest.fixture(scope="module")
def pose_dataset():
    num_past_frames_max = 8
    num_future_frames_max = 5
    num_keypoints = 586
    num_dims_per_keypoint = 3

    dataset = DynamicPosePredictionDataset(
        data_dir='signwriting_animation/data/test_data/pose_samples',
        csv_path='signwriting_animation/data/test_data/transcription_samples/data_subset.csv',
        num_past_frames=num_past_frames_max,
        num_future_frames=num_future_frames_max,
        with_metadata=True,
        split='train'
    )
    return dataset, num_past_frames_max, num_future_frames_max, num_keypoints, num_dims_per_keypoint


def test_get_batch_from_dataset(pose_dataset):
    dataset, num_past_frames_max, num_future_frames_max, num_keypoints, num_dims_per_keypoint = pose_dataset

    batch = dataset[0]

    num_future_frames_sel = batch['data'].shape[0]
    num_past_frames_sel = batch['conditions']['input_pose'].shape[0]

    assert 0 <= num_future_frames_sel <= num_future_frames_max
    assert 0 <= num_past_frames_sel <= num_past_frames_max

    assert batch['data'].shape == (num_future_frames_sel, 1, num_keypoints, num_dims_per_keypoint)
    assert batch['conditions']['input_pose'].shape == (num_past_frames_sel, 1, num_keypoints, num_dims_per_keypoint)
    assert batch['conditions']['input_mask'].shape == (num_past_frames_sel, 1, num_keypoints, num_dims_per_keypoint)
    assert batch['conditions']['target_mask'].shape == (num_future_frames_sel, 1, num_keypoints, num_dims_per_keypoint)
    assert batch['conditions']['sign_image'].shape == (3, 224, 224)

    assert batch['id'] == 'whatsthatsign-524.pose'
