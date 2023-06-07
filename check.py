# commands you need to run on jetson nano 

# git clone https://github.com/ultralytics/yolov5.git
# cd yolov5
# pip install -r requirements.txt

import cv2
import numpy as np

import io

## define the S3 bucket name and key prefix
# bucket_name = 'your-bucket-name'
# key_prefix = 'frames/'

# create an S3 client
# s3_client = boto3.client('s3')

# define the YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).autoshape() 

# define the image transforms
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((640, 640)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# define the video capture object
cap = cv2.VideoCapture(0)

while True:
    # read a frame from the video stream
    ret, frame = cap.read()
    
    # apply the YOLOv5 object detection model to the frame
    # input_img = transform(frame).unsqueeze(0)
    # pred = model(input_img)[0]
    
    # # extract the bounding box coordinates and class labels
    # boxes = pred[:, :4].cpu().numpy()
    # scores = pred[:, 4].cpu().numpy()
    # labels = pred[:, 5].cpu().numpy()
    
    # # filter out low-confidence detections
    # mask = scores > 0.5
    # boxes = boxes[mask]
    # labels = labels[mask]
    
    # frame_number = 0
    # # draw the bounding boxes and class labels on the frame
    # for box, label in zip(boxes, labels):
    #     x1, y1, x2, y2 = box.astype(np.int32)

    # # cropping the image
    # upload_frame = frame[x1 : x2, y1 : y2]
            
    # # upload the frame to S3
    # if ret:
    #     key = key_prefix + 'frame_{}.jpg'.format(frame_number)
    #     frame_number += 1
        
    #     # encode the frame as a JPEG image
    #     success, buffer = cv2.imencode('.jpg', upload_frame)
    #     img_bytes = io.BytesIO(buffer).getvalue()
        
    #     # upload the image to S3
    #     s3_client.put_object(Bucket=bucket_name, Key=key, Body=img_bytes)
    
    # break if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()
