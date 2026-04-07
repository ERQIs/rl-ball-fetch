import traceback
from pathlib import Path

def main():
    try:
        from src.datasets.trajectory_dataset import TrajectoryDataset
        data_root = r"D:\projects\rl-ball-fetch\ball_fetch\vis_backbone\datasets\manual_capture\20260314_140730"
        print('Data root:', data_root)
        ds = TrajectoryDataset(root=data_root,
                               img_size=64,
                               observation_length=16,
                               frame_stride=1,
                               use_last_n_frames=False,
                               observation_end_fraction=0.67,
                               sampling_mode="uniform_visible")
        print('Found episodes:', len(ds))
        if len(ds) == 0:
            print('No episodes found under the data root.')
            return
        sample = ds[0]
        print('Sample keys:', list(sample.keys()))
        print('Frames shape:', getattr(sample['frames'], 'shape', None))
        print('Target:', sample['target_xy'])
        print('Episode id:', sample['episode_id'])
        print('Frame indices:', sample['frame_indices'])
    except Exception as e:
        print('Error during dataset smoke test:')
        traceback.print_exc()

if __name__ == '__main__':
    main()
