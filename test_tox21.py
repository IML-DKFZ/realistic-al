from src.data.data import TorchVisionDM

if __name__ == "__main__":
    data_root = "data"  # Update this with the actual path to your data
    dm = TorchVisionDM(data_root=data_root, dataset="tox21")
    dm.prepare_data()
    dm.setup()

    # Test train dataloader
    train_loader = dm.train_dataloader()
    for features, labels in train_loader:
        print("Train batch features shape:", features.shape)
        print("Train batch labels shape:", labels.shape)
        break

    # Test validation dataloader
    val_loader = dm.val_dataloader()
    for features, labels in val_loader:
        print("Validation batch features shape:", features.shape)
        print("Validation batch labels shape:", labels.shape)
        break
