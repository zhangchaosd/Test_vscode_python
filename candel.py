import cv2

cap = cv2.VideoCapture('/Users/zhangchao/Downloads/reenact_t4_t2.mp4')
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(num_frames)
cnt = 0
for i in range(0, 100):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    flag, frame = cap.read()
    if flag:
        print('succ ',i)
    else:
        print('read fail')

cap.release()