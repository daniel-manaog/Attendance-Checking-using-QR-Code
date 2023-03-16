#Import Libraries
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import requests
import datetime
from PIL import ImageFont, ImageDraw, Image

# Initializing the Model
mpHands = mp.solutions.hands
hands = mpHands.Hands(
	static_image_mode=False,
	model_complexity=1,
	min_detection_confidence=0.95,
	min_tracking_confidence=0.95,
	max_num_hands=2)


#Main CV2 Window
window_name = "Imageality Co Attendance QR Scanner v1.0"
cv2.startWindowThread()
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
cv2.resizeWindow(window_name, 1080, 1920)
cv2.moveWindow(window_name, 1920,0)
##first_read = vid.read()[1]
##center = first_read.shape
##x = center[1]/2 - 1920/2
##y = center[0]/2 - 1080/2

def welcome():
    p2 = threading.Thread(target=detect_palm)
    p2.start()
    fontpath = "fonts/OldSansBlack.ttf"  
    while detected_palm == False:
        img = cv2.imread('img/welcome.png')
        #idleOverlay = cv2.imread("welcome.png", cv2.IMREAD_UNCHANGED)
        font = cv2.FONT_HERSHEY_DUPLEX
        e = datetime.datetime.now()
        #cv2.putText(img, e.strftime("%I:%M:%S %p"), (50,600), font, 2.0, (255, 255, 255), 2)
        #cv2.putText(img, e.strftime("%B %d, %Y"), (50,660), font, 2.0, (255, 255, 255), 2)
        #cv2.putText(img, e.strftime("(%A)"), (50,720), font, 1.8, (255, 255, 255), 2)
   
        font = ImageFont.truetype(fontpath, 50)

        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text((200, 430), e.strftime("%I:%M:%S %p"), font = font)
        draw.text((150, 490), e.strftime("%B %d, %Y"), font = font)
        draw.text((250, 550), e.strftime("(%A)"), font = font)

        img = np.array(img_pil)
        cv2.imshow(window_name, img)
        if cv2.waitKey(200) & 0xFF == 27:
            break


def detect_palm():
    # define a video capture object
    global detected_palm
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
	# Read video frame by frame
        success, img = cap.read()
        # Flip the image(frame)
        img = cv2.flip(img, 1)
        # Convert BGR image to RGB image
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Process the RGB image
        results = hands.process(imgRGB)
        # If hands are present in image(frame)
        if results.multi_hand_landmarks:
                # Both Hands are present in image(frame)
                if len(results.multi_handedness) == 2:
                                # Display 'Both Hands' on the image

                        cap.release()
                        break
                # If any hand present
                else:
                    cap.release()
                    break
##                    for i in results.multi_handedness:
##                            
##                            # Return whether it is Right or Left Hand
##                            print(MessageToDict(i)['classification'][0]['label'])
##                            label = MessageToDict(i)['classification'][0]['label']
##
##                            if label == 'Left':
##                                    
##                                    # Display 'Left Hand' on
##                                    # left side of window
##                                    cv2.putText(img, label+' Hand',
##                                                            (20, 50),
##                                                            cv2.FONT_HERSHEY_COMPLEX,
##                                                            0.9, (0, 255, 0), 2)
##
##                            if label == 'Right':
##                                    
##                                    # Display 'Left Hand'
##                                    # on left side of window
##                                    cv2.putText(img, label+' Hand', (460, 50),
##                                                            cv2.FONT_HERSHEY_COMPLEX,
##                                                            0.9, (0, 255, 0), 2)
##                                    test()
##                                    break

        # is entered, destroy the window
        #cv2.imshow('Image', img)
        if cv2.waitKey(1) & 0xff == ord('q'):
                break
    detected_palm = True

def qr_decode():
    global req_status
    # Overlay Stuff
    imgScan = cv2.imread("img/scan.png", cv2.IMREAD_UNCHANGED)
    # extract alpha channel from foreground image as mask and make 3 channels
    alpha = imgScan[:,:,3]
    alpha = cv2.merge([alpha,alpha,alpha])
    # extract bgr channels from foreground image
    front = imgScan[:,:,0:3]

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    #vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    #vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

    detector = cv2.QRCodeDetector()
    thread_wait = threading.Thread(target=qr_timer)
    thread_wait.start()
    while True:
        # Check the QR Timer
        if qr_wait == 0:
            qr_timed_out()
            break
        # Capture the video frame by frame
        ret, frame = cap.read()
        data, bbox, straight_qrcode = detector.detectAndDecode(frame)
        if len(data) > 0:
            p1 = threading.Thread(target=visit_database, args=(data,))
            p1.start()
            loading_screen()
            print(req_status)
            if req_status == 'Recognized':
                imgRec = cv2.imread('img/qr_recognized.png')
                cv2.imshow(window_name, imgRec)
                cv2.waitKey(3500)
            elif req_status == 'Existing':
                imgRec = cv2.imread('img/qr_existing.png')
                cv2.imshow(window_name, imgRec)
                cv2.waitKey(3500)            
            else:
                imgRec = cv2.imread('img/qr_error.png')
                cv2.imshow(window_name, imgRec)
                cv2.waitKey(6000)
            break

            
        crop_img = frame[int(50):int(480), int(160):int(480)]
        stch_img = cv2.resize(crop_img, (720,1280))
        # blend the two images using the alpha channel as controlling mask
        result = np.where(alpha==(0,0,0), stch_img, front)
        cv2.imshow(window_name, result)


        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def qr_timed_out():
    img = cv2.imread('img/qr_timeout.png')
    cv2.imshow(window_name, img)
    cv2.waitKey(5000)
    
def loading_screen():
    vidLoc = cv2.VideoCapture('img/loading.gif')

    while req_status == '':

        ret, vid = vidLoc.read()
        if not ret:
            vidLoc = cv2.VideoCapture('img/loading.gif')
            continue
        cv2.imshow(window_name, vid)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    return 
        
def visit_database(uuid):
    global req_status

    payload = { 'id': uuid,
                'select': 'uuid'
    }
    response = requests.get('https://'+url+'/attendance-tracker/get_single_data', params=payload)
    time.sleep(1)
    try:
        json_data = response.json()
        print(json_data)
        if json_data['status'] == 200:
                headers = {'Content-Type': 'application/x-www-form-urlencoded'}
                
                payload = { 'id': uuid,
                            'professor': "Engr. Rolito L. Mahaguay",
                            'key': 20201
                }
                print(payload)
                response = requests.post('https://'+url+'/attendance-tracker/post_attendance', headers=headers, data=payload)
                try:
                    json_data = response.json()
                    print(json_data)
                    time.sleep(1)
                    if json_data['status'] == 200:
                        req_status = 'Recognized'
                        return
                    elif json_data['status'] == 300:
                        req_status = 'Existing'
                        return
                    else:
                        req_status = 'Error'
                        return
                    
                except Exception as e:
                    req_status = 'Error'
                    return
                    print(e)
        elif json_data['status'] == 300:
            message = json_data['message']
            if message['uuid'] == uuid:
                req_status = 'Existing'
                return
        req_status = 'Error'
        return #returned as false
            
    except Exception as e:
        req_status = 'Error'
        return
        print(e)
        

def qr_timer():
    global qr_wait
    #Only wait 10 seconds for the QR
    qr_wait = 15
    while qr_wait > 0 and req_status == '':
        time.sleep(1)
        print(f"Waiting for {qr_wait} seconds")
        qr_wait = qr_wait - 1

#qr_timed_out()
url = 'imageality.eu.org'
#visit_database()
#req_status=''

while True:
    qr_wait = 0
    req_status = ''
    detected_palm = False
    welcome()
    #detect_palm()
    qr_decode()
##p1 = threading.Thread(target=welcome)
##p1.start()

#loading_screen()
# After the loop release the cap object
#vid.release()
# Destroy all the windows
#cv2.destroyAllWindows()
