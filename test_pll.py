from tqdm.auto import tqdm
import time


if __name__ == "__main__":
    for epoch in tqdm(range(1, 6), desc="Epochs"):
        for batch in tqdm(range(1, 11), desc="Batches", leave=False):
            time.sleep(0.2)
            tqdm.write(f"Processing batch {batch} of epoch {epoch}...")
        tqdm.write(f"Epoch {epoch} completed.")
    
    print("All epochs completed.")
  