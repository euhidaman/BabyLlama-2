from pathlib import Path
from mrclean import *
from tokenizers import (Tokenizer, decoders, models, pre_tokenizers,
                        processors, trainers)
from tokenizers.normalizers import NFKC


def clean_data():
    DATA_ROOT = Path('..')
    SEQ_LENGTH = 128  # this is a legacy parameter, it does not affect cleaning
    DATA_SPLITS = ['train_10M', 'dev']

    CLEANUP_FUNCTIONS = {
        'childes': cleanup_aochildes,
        'bnc_spoken': cleanup_bnc_spoken,
        'cbt': cleanup_cbt,
        'children_stories': cleanup_children_stories,
        'gutenberg': cleanup_gutenberg,
        'open_subtitles': cleanup_open_subtitles,
        'qed': cleanup_qed,
        'simple_wiki': cleanup_simple_wikipedia,
        'switchboard': cleanup_switchboard,
        'wikipedia': cleanup_wikipedia,
    }

    for split in DATA_SPLITS:
        INPUT_DIR = DATA_ROOT / 'data' / split
        OUTPUT_DIR = DATA_ROOT / 'data' / f'{split}_clean'

        OUTPUT_DIR.mkdir(exist_ok=True)

        train_files = [f for f in INPUT_DIR.iterdir() if f.is_file() and f.suffix in [
            '.train', '.dev']]

        for file in train_files:
            text = file.read_text()
            cleaned_text = CLEANUP_FUNCTIONS[file.stem](text, SEQ_LENGTH)
            (OUTPUT_DIR / file.name).write_text(cleaned_text)
            print(
                f"🧹 Cleaned '{file.name}' (size {len(text)} -> {len(cleaned_text)}) in {split}")


def train_tokenizer():
    DATA_ROOT = Path('..')
    # We train the tokenizer on the train data only
    data_dir = DATA_ROOT / 'data' / 'train_10M_clean'

    paths = [str(f) for f in data_dir.glob('*') if f.is_file()
             and not f.name.endswith('.DS_Store') and f.suffix in ['.train']]

    print(f"Found {len(paths)} training files")
    assert len(paths) > 0, 'No data files found'

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    tokenizer.normalizer = NFKC()

    trainer = trainers.BpeTrainer(
        vocab_size=16000, min_frequency=2, special_tokens=["<pad>", "<s>", "</s>"])
    tokenizer.train(paths, trainer)

    tokenizer_path = DATA_ROOT / 'models/gpt-clean-16000.json'
    (DATA_ROOT / 'models').mkdir(exist_ok=True)
    tokenizer.save(str(tokenizer_path), pretty=True)
    print(f"Saved tokenizer to {tokenizer_path}")

    # Test the tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    test_text = "The quick brown fox jumps over the lazy dog."
    encoded = tokenizer.encode(test_text)
    print(f"\nTesting tokenizer:")
    print(f"Encoded String: {encoded.tokens}")
    print(f"Encoded IDs: {encoded.ids}")
    decoded = tokenizer.decode(encoded.ids)
    print(f"Decoded String: {decoded}")


if __name__ == "__main__":
    print("Starting data cleaning...")
    clean_data()
    print("\nStarting tokenizer training...")
    train_tokenizer()
