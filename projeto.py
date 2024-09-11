import cv2
import mediapipe as mp
import numpy as np
import time

### Tipagem =====================================
confidence = float
webcam_image = np.ndarray
rgb_tuple = tuple[int,int,int]
# Coords_ vector


### Classe =================================
class Detector:
    def __init__(self,
                 mode: bool = False,
                 number_hands: int = 2,
                 model_complexity: int = 1,
                 min_detec_confidence: confidence = 0.5,
                 min_tracking_confidence: confidence =0.5):
        #parametros necessarios para iniciar
        self.mode = mode
        self.max_num_hands = number_hands
        self.complexity = model_complexity
        self.detection_con = min_detec_confidence
        self.tracking_con = min_tracking_confidence

        # iniciar o hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            self.mode,
            self.max_num_hands,
            self.complexity,
            self.detection_con,
            self.tracking_con  )
        self.mp_draw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]

    def find_hands(self,
                   img: webcam_image,
                   draw_hands: bool = True):
        # Correção de cor 
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # coletar resultados dos processos das hands e analisar
        self.results = self.hands.process(img_RGB)
        if self.results.multi_hand_landmarks and draw_hands:
            for hand in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand, self.mp_hands.HAND_CONNECTIONS)
        return img





### TESTE DE CLASSE ==========
if __name__ == '__main__':
    #classe
    Detec = Detector()

    # captura de imagem
    capture = cv2.VideoCapture(0)
    while True:
        # Captura do frame / imagem
        _, img = capture.read()

        # manipulação de frame
        img = Detec.find_hands(img)


        cv2.imshow('Camera do notebook', img)
        # quitando
        if cv2.waitKey(20) & 0xFF==ord('q'):
            break 