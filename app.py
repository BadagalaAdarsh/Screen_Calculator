import mediapipe as mp
import cv2
from mediapipe.python.solutions.drawing_styles import _INDEX_FINGER_LANDMARKS, _THUMP_LANDMARKS
import numpy as np



class Button:

    def __init__(self, pos, width, height, value):

        self.pos = pos 
        self.width = width
        self.height = height 
        self.value = value 

        # cv2.rectangle(img, (100,180), (180,260),(225, 225, 225), cv2.FILLED)
        # cv2.rectangle(img, (100,180), (180,260),(50, 50, 50), 3)
        # cv2.putText(img, "9", (100 + 30, 180 + 50), cv2.FONT_HERSHEY_PLAIN, 3, (50, 50, 50), 3)
        # button1 = Button((900, 180), 80, 80,"5")

    def draw (self, img):

        cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height) ,(0, 225, 0), 1)
        cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height) ,(0, 0, 0), 3 )
        cv2.putText(img, self.value, (self.pos[0] + 30, self.pos[1] + 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)


    def checkClick(self, x, y):

        if self.pos[0] < x < self.pos[0] + self.width and self.pos[1] < y < self.pos[1] + self.height:

            cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height) ,(255, 255, 255), cv2.FILLED)
            cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height) ,(0, 0, 0), 3 )
            cv2.putText(img, self.value, (self.pos[0] + 15, self.pos[1] + 60), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 5)

            return True 

        else:
            return False

    

def findDistance(x, y, a, b):

    result = (((( x - a) **2) + ((y - b) ** 2)) ** 0.5)
    return int(result)


cap = cv2.VideoCapture(0)
cap.set(3, 1280) # width
cap.set(4, 1280) # height

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Creating Buttons
buttonListValues = [ ["7", "8", "9", "*"],
                     ["4", "5", "6", "-"],
                     ["1", "2", "3", "+"],
                     ["0", "/", ".", "="]]


buttonList = [] 

for x in range(4):
    for y in range(4):
        xpos = x * 80 + 850
        ypos = y * 80 + 250
        buttonList.append(Button((xpos, ypos), 80, 80, buttonListValues[y][x]))


# variables 
myEquation = ""
delayCounter = 0



with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands = 1) as hands: 
    while cap.isOpened():

        suceess, img = cap.read()      
        
        
        # BGR 2 RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Flip on horizontal
        img = cv2.flip(img, 1)
        
        # Set flag
        img.flags.writeable = False
        
        # Detections
        results = hands.process(img)
        
        # Set flag to true
        img.flags.writeable = True
        
        # RGB 2 BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        imageHeight , imageWidth , _ = img.shape
        
               

        # Draw all buttons
        cv2.rectangle(img, (850, 170), (850 + 320, 170 +70) ,(0, 225, 225), 1)
        cv2.rectangle(img, (850, 170), (850 + 320, 170 +70) , (0, 0, 0), 3 )



        # rendering buttons
        for button in buttonList:
            button.draw(img)

        distance = float('inf')
    
        if results.multi_hand_landmarks:
                          

            for handLandmarks in results.multi_hand_landmarks:

                count = 0
                for point in mp_hands.HandLandmark:

                    normalizedLandmark = handLandmarks.landmark[point]
                    pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x , normalizedLandmark.y, imageWidth, imageHeight)

                    if count == 8:
                        normalizedLandmark = handLandmarks.landmark[point]
                        pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, imageWidth, imageHeight)

                        index_x = pixelCoordinatesLandmark[0]
                        index_y = pixelCoordinatesLandmark[1]

                    if count == 12:
                        normalizedLandmark = handLandmarks.landmark[point]
                        pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, imageWidth, imageHeight)

                        middle_x = pixelCoordinatesLandmark[0]
                        middle_y = pixelCoordinatesLandmark[1]
                        break


                    count += 1

            img = cv2.circle(img, (index_x, index_y), radius=3, color = (0, 0, 255), thickness=5)

            distance = findDistance(index_x, index_y, middle_x, middle_y)
            

            if distance < 35:
                for i, button in enumerate(buttonList):
                    if button.checkClick(index_x, index_y) and delayCounter == 0:
                        myValue = buttonListValues[i%4][i//4]

                        if myValue == "=":
                            myEquation = str(eval(myEquation))
                        else:
                            myEquation += myValue


                        delayCounter = 1

        
        # to avoid duplicates

        if delayCounter != 0:
            delayCounter += 1 

            if delayCounter > 10:
                delayCounter = 0




        cv2.putText(img, myEquation, (850+30, 170 + 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 3)
            
        
        cv2.imshow('Hand Tracking', img)

        key = cv2.waitKey(1)

        if key == ord('c'):
            myEquation = ""

        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()