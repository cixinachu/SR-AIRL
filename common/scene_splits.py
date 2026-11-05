"""Single source of truth for non-overlapping NGSIM experiment splits."""
TRAIN_SCENE_IDS = tuple(range(0, 100))
VAL_SCENE_IDS = tuple(range(100, 120))
TEST_SCENE_IDS = tuple(range(120, 150))

SCENE_SPLITS = {'train': TRAIN_SCENE_IDS, 'val': VAL_SCENE_IDS, 'test': TEST_SCENE_IDS}


def split_scene_ids(split):
    try:
        return SCENE_SPLITS[split]
    except KeyError as exc:
        raise ValueError(f'Unknown split: {split}') from exc

