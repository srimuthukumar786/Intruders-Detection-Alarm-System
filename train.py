from ultralytics import YOLO

model = YOLO('yolov8n.pt')

def main():
    model.train(data='C:/Users/Muthukumar MSc/PycharmProjects/Antispoofing/Dataset/SplitData/dataoffline.yaml', epochs=3)


if __name__ == '__main__':
    main()