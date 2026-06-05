# from ultralytics import YOLO

# # Load a pretrained YOLO26n model
# model = YOLO("yolov11n_last.pt")

# # Define path to video file
# source = "C:/Users/admin/Developer/DeepSORT_test/UA_Detrac_test.mp4"

# # Run inference on the source
# results = model(source, stream=True, show=True, save=True)  # generator of Results objects

# for r in results:
#     # Đoạn pass này tạm thời để trống, show=True sẽ tự lo việc vẽ ảnh lên màn hình.
#     pass

from ultralytics import YOLO

# Thay bằng đường dẫn đến file weights bạn vừa train (thường ở dạng runs/detect/train/weights/best.pt)
model = YOLO("C:/Users/admin/Developer/DeepSORT_test/yolov11n_best (1).pt")

# In ra toàn bộ danh sách ID và Tên class
print(model.names)