from signwriting_animation.data.data_loader import DynamicPosePredictionDataset

def test_get_batch():
    num_past_frames_max = 40
    num_future_frames_max = 20
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
