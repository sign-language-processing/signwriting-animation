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
        with_metadata=False,
        split='train'
    )
    return dataset, num_past_frames_max, num_future_frames_max, num_keypoints, num_dims_per_keypoint


def test_get_sample_from_dataset(pose_dataset):
    dataset, num_past_frames_max, num_future_frames_max, num_keypoints, num_dims_per_keypoint = pose_dataset

    sample = dataset[0]

    num_future_frames_sel = len(sample['data'])
    num_past_frames_sel = len(sample['conditions']['input_pose'])

    pose_frame_shape = (1, num_keypoints, num_dims_per_keypoint)

    assert 0 <= num_future_frames_sel <= num_future_frames_max
    assert 0 <= num_past_frames_sel <= num_past_frames_max

    assert sample['data'].shape == (num_future_frames_sel, *pose_frame_shape)
    assert sample['conditions']['input_pose'].shape == (num_past_frames_sel, *pose_frame_shape)
    assert sample['conditions']['input_mask'].shape == (num_past_frames_sel, *pose_frame_shape)
    assert sample['conditions']['target_mask'].shape == (num_future_frames_sel, *pose_frame_shape)
    assert sample['conditions']['sign_image'].shape == (3, 224, 224)

    assert sample['id'] == 'whatsthatsign-524.pose'
