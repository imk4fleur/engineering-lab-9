"""Генерация учебного CSV порциями, без хранения всего файла в памяти."""
import argparse
from app import demo_csv

parser = argparse.ArgumentParser(description='Создать синтетический CSV для практической №9')
parser.add_argument('path', help='путь к новому CSV')
parser.add_argument('--rows', type=int, default=1_000_000)
args = parser.parse_args()
if args.rows < 10:
    parser.error('нужно минимум 10 строк')
with open(args.path, 'x', encoding='utf-8', newline='') as output:
    for offset in range(0, args.rows, 10_000):
        batch = demo_csv(min(10_000, args.rows - offset), seed=42 + offset)
        output.write(batch if offset == 0 else batch.split('\n', 1)[1])
print(f'Создан файл {args.path}, строк: {args.rows}')
