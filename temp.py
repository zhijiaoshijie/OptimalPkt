import time
import sys

def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()

total_items = 10

for i in range(total_items):
    print_progress_bar(i, total_items, prefix='Progress:', suffix='Complete', length=50)
    print(f'\nProcessing item {i+1}/{total_items}')
    time.sleep(0.5)  # Simulate some processing time

# When everything's done
print_progress_bar(total_items, total_items, prefix='Progress:', suffix='Complete', length=50)
print("\nDone!")

