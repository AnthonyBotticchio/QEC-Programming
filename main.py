from model import training

DATA_DIR = "../data"

def main():
    print("Starting main...")

    training.train_ev_usage_model(DATA_DIR)

if __name__ == "__main__":
    main