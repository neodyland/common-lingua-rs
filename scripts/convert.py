import torch


def main():
    checkpoint = torch.load("./model.pt", map_location="cpu", weights_only=False)
    for name, tensor in checkpoint["model_state_dict"].items():
        with open(f"../common-lingua/src/weights/{name}.bin", "wb") as f:
            f.write(tensor.numpy().tobytes())
        with open(f"../common-lingua/src/weights/{name}.size", "w") as f:
            f.write(",".join([str(m) for m in list(tensor.shape)]))


if __name__ == "__main__":
    main()
