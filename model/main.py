import training as Training

DATA_DIR = "../data"

def main():
    print("Starting main...")

    Training.train_ev_usage_model(DATA_DIR)

if __name__ == "__main__":
    main