import cv2
import numpy as np
from TrainingCNNFunction import forwardInterface, loadWeights, NonMaximumSupression, preprocess, FPSandShow
from training1 import loadWeight, model

kernels, bias, gammas, betas, running_means, running_vars, anchorbase, nc = loadWeights("Weight/weights.npz")
kernelsNhandien, weightsNhandien = loadWeight()
url_str = "video/video.mp4"
cap = cv2.VideoCapture(url_str)
listDT = ["Lam", "Nghia", "Nguyen", "Quang"]
listCheckDT = [False, False, False, False]

def drawBBx(input, img):
    rs = input[0]
    for bbx, conf, cls in rs:
        x1, y1, x2, y2 = bbx
        center_x, center_y = int(x1+(x2-x1)/2), int(y1+(y2-y1)/2)
        if conf >= 0.35:
            cv2.circle(img, (center_x, center_y), 2, (0, 255, 0), -1)
            image = cv2.resize(img[y1:y2, x1:x2], (32, 32))
            rs = np.argmax(model(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), kernelsNhandien, weightsNhandien[0], weightsNhandien[1], "gray"))
            listCheckDT[rs] = True
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(img, f"{listDT[rs]} {int(conf*100)/100}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
    return img

if __name__ == "__main__":
    while True:
        ret, frame = cap.read()
        # ret, frame = True, img.copy()
        if ret:
            FPSandShow(frame)
            frame = cv2.resize(frame, (640, 480))
            frame1 = preprocess(frame)
            outobjF1, outobjF2, outclsF1, outclsF2, outbbxF1, outbbxF2 = forwardInterface(frame1, kernels, bias, gammas, betas, running_means, running_vars, anchorbase, nc)
            outNMS = NonMaximumSupression([outbbxF1, outbbxF2], [outclsF1, outclsF2], [outobjF1, outobjF2], 0.2)
            for i, name in enumerate(listDT):
                strDT = "Chua diem danh"
                if listCheckDT[i] == True:
                    strDT = "Da diem danh"
                cv2.putText(frame, f"{name}: {strDT}", (450, (i + 1) * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(frame, f"{name}: {strDT}", (450, (i+1)*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            frame = drawBBx(outNMS, frame)
            cv2.imshow("cv2", frame)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
    # cap.release()
    cv2.destroyAllWindows()
