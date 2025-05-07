from pathlib import Path


def split_file(path: str, train_dir='../train_9.5M', test_dir='../select_0.5M', test_fraction: float = 0.05):
    """Splits a file into two train/test files, by lines"""
    path = Path(path)
    text = path.read_text()
    lines = text.splitlines()
    N_train_lines = round((1-test_fraction)*len(lines))
    train_text = '\n'.join(lines[:N_train_lines])
    test_text = '\n'.join(lines[N_train_lines:])
    (path.parent / train_dir).mkdir(exist_ok=True)
    (path.parent / test_dir).mkdir(exist_ok=True)
    (path.parent / train_dir / path.name).write_text(train_text)
    (path.parent / test_dir / path.name).write_text(test_text)
    return len(lines), N_train_lines


def main():
    DATA = Path('../data')

    print("Starting dataset splitting...")
    total_lines = 0
    total_train_lines = 0

    for file in (DATA / 'train_10M_clean').glob('*.train'):
        n_lines, n_train = split_file(file)
        total_lines += n_lines
        total_train_lines += n_train
        print(f"Split {file.name}: {n_train}/{n_lines} lines for training")

    print(f"\nTotal: {total_train_lines}/{total_lines} lines for training")
    print("Done splitting datasets")


if __name__ == "__main__":
    main()
