import argparse
import torch
import numpy as np
import cv2
import os
import time

from utils.misc import load_weight
from data.voc0712 import VOCDetection, VOC_CLASSES
from data.transform import BaseTransform

from models.build import build_yolo
from rail.pipeline import processing
from rail import line
from rail import railutils
import pdb


parser = argparse.ArgumentParser(description='YOLO Detection')
parser.add_argument('-d', '--dataset', default='custom',
                    help='voc, coco-val custom.')
parser.add_argument('--root', default='./',
                    help='data root')
# parser.add_argument('-size', '--input_size', default=416, type=int,
#                     help='输入图像尺寸')
parser.add_argument('-size', '--input_size', default=608, type=int,
                    help='输入图像尺寸')

parser.add_argument('-v', '--version', default='yolo',
                    help='yolo')
parser.add_argument('--weight', default='./weights/voc/yolo/yolo_epoch_121_69.1.pth',
                    type=str, help='模型权重的路径')
parser.add_argument('--conf_thresh', default=0.1, type=float,
                    help='得分阈值')
parser.add_argument('--nms_thresh', default=0.50, type=float,
                    help='NMS 阈值')
parser.add_argument('-vs', '--visual_threshold', default=0.5, type=float,
                    help='用于可视化的阈值参数')
parser.add_argument('--cuda', action='store_true', default=False, 
                    help='use cuda.')
parser.add_argument('--save', action='store_true', default=True, 
                    help='save vis results.')

args = parser.parse_args()


def plot_bbox_labels(img, bbox, label=None, cls_color=None, text_scale=0.4):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    # plot bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)
    
    if label is not None:
        # plot title bbox
        cv2.rectangle(img, (x1, y1-t_size[1]), (int(x1 + t_size[0] * text_scale), y1), cls_color, -1)
        # put the test on the title bbox
        cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, text_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return img


def visualize_rail(img, 
              bboxes, 
              scores, 
              labels, 
              vis_thresh, 
              class_colors, 
              class_names, 
              class_indexs=None, 
              dataset_name='voc',
              rail_area=None,
              last_frame=0,
              now_frame=0):
    ts = 0.4
    safe = True
    dangerous_frame = last_frame
    for i, bbox in enumerate(bboxes):
        if scores[i] > vis_thresh:
            cls_id = int(labels[i])
            if dataset_name == 'coco-val':
                cls_color = class_colors[cls_id]
                cls_id = class_indexs[cls_id]
            else:
                cls_color = class_colors[cls_id]
                
            if class_names[cls_id]!='person':
                continue
            
            if len(class_names) > 1 :
                mess = '%s: %.2f' % (class_names[cls_id], scores[i])
                x1, y1, x2, y2 = bbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # print(x1, y1, x2, y2)
                if x1<960//2 and x2<960//2:
                    continue
                try:
                    if rail_area[y1][x1] | rail_area[y2][x1] | rail_area[y1][x2] |rail_area[y2][x2]:
                        safe = False
                        dangerous_frame = now_frame
                except:
                    pass
            else:
                cls_color = [255, 0, 0]
                mess = None
            img = plot_bbox_labels(img, bbox, mess, cls_color, text_scale=ts)
    
    if safe and now_frame-last_frame>=15:
        cv2.putText(img, "Safe!", (40, 60), 0, 2, (0, 0, 255), 2, lineType=cv2.LINE_AA)
    else:
        cv2.putText(img, "Dangerous!", (40,60), 0, 2, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        
    return img, dangerous_frame
        

def detect_rail(args, model, device, img, transform, class_colors=None, class_names=None, 
                class_indexs=None, rail_area=None, dangerous_frame=0, now_frame=0):
    
    h, w, _ = img.shape
    # print(h, w)
    
    # 预处理图像，并将其转换为tensor类型
    x = torch.from_numpy(transform(img)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
    x = x.unsqueeze(0).to(device)

    t0 = time.time()
    # 前向推理
    bboxes, scores, labels = model(x)
    print("detection time used ", time.time() - t0, "s")
    
    # 将预测的输出映射到原图的尺寸上去
    scale = np.array([[w, h, w, h]])
    bboxes *= scale

    # 可视化检测结果
    img_processed, new_frame = visualize_rail(
        img=img,
        bboxes=bboxes,
        scores=scores,
        labels=labels,
        vis_thresh=args.visual_threshold,
        class_colors=class_colors,
        class_names=class_names,
        class_indexs=class_indexs,
        dataset_name=args.dataset,
        rail_area=rail_area,
        last_frame=dangerous_frame,
        now_frame=now_frame
        )
    
    return img_processed, new_frame
        
        
        

if __name__ == '__main__':
    # 是否使用cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 输入图像的尺寸
    input_size = args.input_size

    class_names = VOC_CLASSES
    class_indexs = None
    num_classes = 20
    
    # 用于可视化，给不同类别的边界框赋予不同的颜色，为了便于区分。
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(num_classes)]

    # 构建模型
    model = build_yolo(args, device, input_size, num_classes, trainable=False)

    # 加载已训练好的模型权重
    model = load_weight(model, args.weight)
    model.to(device).eval()
    print('Finished loading model!')

    val_transform = BaseTransform(input_size)


    cap = cv2.VideoCapture('./Rail.mp4')

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # 获取视频的宽度和高度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 获取视频的帧率

    print(width,height,fps)
    # 创建一个VideoWriter对象来保存视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4格式
    out = cv2.VideoWriter('Rail_detected.mp4', fourcc, fps, (width, height))

    dangerous_frame = -1
    frame_num = 0
    
    # 读取视频直到结束
    while True:
        # 读取下一帧
        ret, frame = cap.read()

        # 如果正确读取帧，ret为True
        if not ret:
            print("Can't receive frame any. Exiting ...")
            break
        
        frame_num += 1
        
        left_line = line.Line()
        right_line = line.Line()
        M,Minv = railutils.get_M_Minv()
        # 进行图像预处理，获取检测火车轨道面积图像
        rail_area = processing(frame,M,Minv,left_line,right_line)
        # print(rail_area.shape)
        # pdb.set_trace()
        # 对视频进行逐帧检测和分析，检测是否有行人在火车轨道线上
        output_frame, dangerous_frame = detect_rail(args=args,
            model=model, 
            device=device, 
            img=frame,
            transform=val_transform,
            class_colors=class_colors,
            class_names=class_names,
            class_indexs=class_indexs,
            rail_area=rail_area,
            dangerous_frame=dangerous_frame,
            now_frame=frame_num
            )
        # 将处理后的帧写入输出视频
        out.write(output_frame)

    # 释放VideoCapture和VideoWriter对象
    cap.release()
    out.release()

    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()
